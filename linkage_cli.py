#!/usr/bin/env python3
"""
Run the privacy evaluation with respect to the risk of linkability.

-----
I edited this script mainly by adding comments during its review and formatting
it to respect pythonic standards.
Nampoina Andriamilanto <tompo.andri@gmail.>
"""

import json
from argparse import ArgumentParser
from pathlib import Path
from warnings import simplefilter

from loguru import logger
from numpy.random import choice, seed
from pandas import DataFrame

from attack_models.mia_classifier import (
    MIAttackClassifierRandomForest, generate_mia_shadow_data,
    generate_mia_anon_data)
from feature_sets.independent_histograms import HistogramFeatureSet
from feature_sets.model_agnostic import NaiveFeatureSet, EnsembleFeatureSet
from feature_sets.bayes import CorrelationsFeatureSet
from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.data_synthesiser import (
    IndependentHistogram, BayesianNet, PrivBayes)
from sanitisation_techniques.sanitiser import SanitiserNHS
from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.constants import LABEL_IN, LABEL_OUT


simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

# The numpy pseudorandom generator seed is specified manually, but you can use
# the random or secrets python library to generate a random seed
SEED = 42


def main():
    """Run the privacy evaluation with respect to the risk of linkability."""
    # Parse the arguments
    argparser = ArgumentParser()
    datasource = argparser.add_mutually_exclusive_group()
    datasource.add_argument(
        '--s3name', '-S3', type=str, choices=['adult', 'census', 'credit',
                                              'alarm', 'insurance'],
        help='Name of the dataset to run on')
    datasource.add_argument('--datapath', '-D', type=str,
                            help='Path to a local data file')
    argparser.add_argument(
        '--runconfig', '-RC', default='runconfig_mia.json', type=str,
        help='Path to the runconfig file')
    argparser.add_argument(
        '--outdir', '-O', default='tests', type=str, help='Path relative to '
        'CWD for storing output files')
    args = argparser.parse_args()

    # Load the run configuration json file
    with open(args.runconfig) as run_config_file:
        runconfig = json.load(run_config_file)
    logger.info(f'Runconfig:\n{runconfig}')

    # Load the dataset
    if args.s3name:
        raw_pop, metadata = load_s3_data_as_df(args.s3name)
        data_name = args.s3name
    else:
        data_path = Path(args.datapath)
        raw_pop, metadata = load_local_data_as_df(data_path)
        data_name = data_path.name
    logger.info(f'Loaded data {data_name}:\n{raw_pop.info()}')

    # Make sure the output directory exists
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Seed the numpy pseudorandom generator
    seed(SEED)

    # ============================== GAME INPUTS ==============================
    # Pick the id of the targets
    target_ids = choice(list(raw_pop.index), size=runconfig['nTargets'],
                        replace=False).tolist()

    # If specified: add specific target ids
    if runconfig['Targets']:
        target_ids.extend(runconfig['Targets'])

    # Get the actual targets
    targets = raw_pop.loc[target_ids, :]

    # Drop targets from the population
    raw_pop_without_targets = raw_pop.drop(target_ids)

    # Init adversary's prior knowledge
    raw_adv_idx = choice(list(raw_pop_without_targets.index),
                         size=runconfig['sizeRawA'], replace=False).tolist()
    raw_adv = raw_pop.loc[raw_adv_idx, :]

    # The list of the candidate generative models to evaluate
    # Format: [ instantiated generative models ]
    gen_model_list = []
    for gen_mod, params_list in runconfig.get('generativeModels', dict()
                                              ).items():
        if gen_mod == 'IndependentHistogram':
            for params in params_list:
                gen_model_list.append(IndependentHistogram(metadata, *params))
        elif gen_mod == 'BayesianNet':
            for params in params_list:
                gen_model_list.append(BayesianNet(metadata, *params))
        elif gen_mod == 'PrivBayes':
            for params in params_list:
                gen_model_list.append(PrivBayes(metadata, *params))
        elif gen_mod == 'CTGAN':
            for params in params_list:
                gen_model_list.append(CTGAN(metadata, *params))
        elif gen_mod == 'PATEGAN':
            for params in params_list:
                gen_model_list.append(PATEGAN(metadata, *params))
        else:
            raise ValueError(f'Unknown generative model {gen_mod}')

    # The list of the candidate sanitisation techniques to evaluate
    # Format: [ instantiated sanitisation techniques ]
    san_tech_list = []
    for name, params_list in runconfig.get('sanitisationTechniques', dict()
                                           ).items():
        if name == 'SanitiserNHS':
            for params in params_list:
                san_tech_list.append(SanitiserNHS(metadata, *params))
        else:
            raise ValueError(f'Unknown sanitisation technique {name}')

    # ============================ ATTACK TRAINING ============================
    logger.info('\n---- attack training ----')

    # Dictionary of the attack models
    # Format: {target id => {
    #              sanitization technique or generative model name => {
    #                  feature extractor name => the actual classifier }}}
    # Given the triplet (target id, privacy mechanism, feature extractor), we
    # can retrieve the classifier that infers whether the target is in the raw
    # dataset
    attacks = {}

    # For each selected target
    for tid in target_ids:
        logger.info(f'\n--- Adversary picks target {tid} ---')
        target = targets.loc[[tid]]
        attacks[tid] = {}

        # For each sanitization technique, train classifiers to recognize
        # whether the target is in the raw dataset
        for san_tech in san_tech_list:
            logger.info(f'Start: attack training for {san_tech.__name__}...')

            attacks[tid][san_tech.__name__] = {}

            # Generate samples to train the attack classifier on
            san_adv, labels_adv = generate_mia_anon_data(
                san_tech, target, raw_adv, runconfig['sizeRawT'],
                runconfig['nShadows'] * runconfig['nSynA'])

            # Train the attack on shadow data for the four feature extractors
            for feature in [NaiveFeatureSet(DataFrame),
                            HistogramFeatureSet(
                                DataFrame, metadata,
                                nbins=san_tech.histogram_size,
                                quids=san_tech.quids),
                            CorrelationsFeatureSet(
                                DataFrame, metadata, quids=san_tech.quids),
                            EnsembleFeatureSet(
                                DataFrame, metadata,
                                nbins=san_tech.histogram_size,
                                quasi_id_cols=san_tech.quids)]:
                attack = MIAttackClassifierRandomForest(
                    metadata=metadata, FeatureSet=feature, quids=san_tech.quids
                    )
                attack.train(san_adv, labels_adv)
                attacks[tid][san_tech.__name__][feature.__name__] = attack

            # Clean up
            del san_adv, labels_adv

            logger.info('Finished: attack training.')

        # For each generative model, train classifiers to recognize whether the
        # target is in the raw dataset
        for gen_model in gen_model_list:
            logger.info(f'Start: attack training for {gen_model.__name__}...')

            attacks[tid][gen_model.__name__] = {}

            # Generate shadow model data for training attacks on this target
            syn_adv, labels_sa = generate_mia_shadow_data(
                gen_model, target, raw_adv, runconfig['sizeRawT'],
                runconfig['sizeSynT'], runconfig['nShadows'],
                runconfig['nSynA'])

            # Train the attack on shadow data for the four feature extractors
            for feature in [NaiveFeatureSet(gen_model.datatype),
                            HistogramFeatureSet(gen_model.datatype, metadata),
                            CorrelationsFeatureSet(gen_model.datatype,
                                                   metadata),
                            EnsembleFeatureSet(gen_model.datatype, metadata)]:
                attack = MIAttackClassifierRandomForest(metadata, feature)
                attack.train(syn_adv, labels_sa)
                attacks[tid][gen_model.__name__][feature.__name__] = attack

            # Clean up
            del syn_adv, labels_sa

            logger.info('Finished: attack training.')

    # ============================== EVALUATION ===============================
    # Format: {target id => { privacy mechanism name => {
    #              iteration number => { feature extractor => {
    #                  'Secret' => [ the targets presence as 0 or 1 ],
    #                  'AttackerGuess' => [ the attacker guesses (0 or 1) ]
    #          }}}}}
    results_target_privacy = {
        tid: {privacy_model.__name__: {}
              for privacy_model in gen_model_list + san_tech_list}
        for tid in target_ids}

    logger.info('\n---- Start the game ----')

    # Run the specified number of iterations
    for n_iter in range(runconfig['nIter']):
        logger.info(f'\n--- Game iteration {n_iter + 1} ---')

        # Draw a raw dataset that does not contain the targets
        r_idx = choice(list(raw_pop_without_targets.index),
                       size=runconfig['sizeRawT'], replace=False).tolist()
        raw_t_out = raw_pop_without_targets.loc[r_idx]

        # Evaluate every generative model
        for gen_model in gen_model_list:
            logger.info(f'Start: Evaluation for model {gen_model.__name__}...')

            # Train the generative model on the raw dataset and generate nSynT
            # synthetic datasets with the s_t = 0 label (i.e., the target is
            # not in the raw dataset)
            gen_model.fit(raw_t_out)
            syn_t_without_target = [
                gen_model.generate_samples(runconfig['sizeSynT'])
                for _ in range(runconfig['nSynT'])]
            syn_labels_out = [LABEL_OUT for _ in range(runconfig['nSynT'])]

            # For each target
            for tid in target_ids:
                logger.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                results_target_privacy[tid][gen_model.__name__][n_iter] = {}

                # Train the generative model on the raw dataset in which we add
                # the target. Then, generate nSynT synthetic datasets with the
                # s_t = 1 label (i.e., the target is in the raw dataset)
                raw_t_in = raw_t_out.append(target)
                gen_model.fit(raw_t_in)
                syn_t_with_target = [
                    gen_model.generate_samples(runconfig['sizeSynT'])
                    for _ in range(runconfig['nSynT'])]
                syn_labels_in = [LABEL_IN for _ in range(runconfig['nSynT'])]

                # The complete dataset on which we train the classifier that
                # infers whether the target is in the raw dataset contains:
                # - The synthetic dataset that misses the target
                # - The synthetic dataset that contains the target
                # - Both types get their corresponding labels in syn_t_labels
                syn_t = syn_t_without_target + syn_t_with_target
                syn_t_labels = syn_labels_out + syn_labels_in

                # Run the attacks
                for feature, attack in attacks[tid][gen_model.__name__
                                                    ].items():
                    # Produce a guess for each synthetic dataset and store it
                    attacker_guesses = attack.attack(syn_t)
                    result_dict = {'Secret': syn_t_labels,
                                   'AttackerGuess': attacker_guesses}
                    results_target_privacy[tid][gen_model.__name__][n_iter][
                        feature] = result_dict

            del syn_t, syn_t_without_target, syn_t_with_target

            logger.info(
                f'Finished: Evaluation for model {gen_model.__name__}.')

        # Evaluate every sanitization technique
        for san_tech in san_tech_list:
            logger.info('Start: Evaluation for sanitiser '
                        f'{san_tech.__name__}...')

            # Sanitize the dataset that does not contain the targets
            san_out = san_tech.sanitise(raw_t_out)

            # For each target
            for tid in target_ids:
                logger.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                results_target_privacy[tid][san_tech.__name__][n_iter] = {}

                # Sanitize the dataset that contains the targets
                raw_t_in = raw_t_out.append(target)
                san_in = san_tech.sanitise(raw_t_in)

                # The complete dataset on which we train the classifier that
                # infers whether the target is in the raw dataset contains:
                # - The sanitized dataset that misses the target
                # - The sanitized dataset that contains the target
                # - Both types get their corresponding labels in san_t_labels
                san_t = [san_out, san_in]
                san_t_labels = [LABEL_OUT, LABEL_IN]

                # Run the attacks
                for feature, attack in attacks[tid][san_tech.__name__].items():
                    # Produce a guess for each sanitized dataset and store it
                    attacker_guesses = attack.attack(
                        san_t, attemptLinkage=True, target=target)
                    result_dict = {'Secret': san_t_labels,
                                   'AttackerGuess': attacker_guesses}
                    results_target_privacy[tid][san_tech.__name__][n_iter][
                        feature] = result_dict

            del san_t, san_out, san_in

            logger.info(f'Finished: Evaluation for model {san_tech.__name__}.')

    # Write the results in the json output file
    output_path = Path(args.outdir) / Path(f'ResultsMIA_{data_name}.json')
    logger.info(f'Write results to {output_path}')
    with open(output_path, 'w+') as output_file:
        json.dump(results_target_privacy, output_file, indent=2,
                  default=json_numpy_serialzer)


if __name__ == "__main__":
    main()
