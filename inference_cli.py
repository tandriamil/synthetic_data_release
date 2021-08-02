#!/usr/bin/env python3
"""
Run the privacy evaluation under an attribute inference adversary.

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

from generative_models.ctgan import CTGAN
from generative_models.pate_gan import PATEGAN
from generative_models.data_synthesiser import (
    IndependentHistogram, BayesianNet, PrivBayes)
from sanitisation_techniques.sanitiser import SanitiserNHS
from attack_models.reconstruction import LinRegAttack, RandForestAttack
from utils.datagen import load_s3_data_as_df, load_local_data_as_df
from utils.utils import json_numpy_serialzer
from utils.constants import LABEL_IN, LABEL_OUT

simplefilter('ignore', category=FutureWarning)
simplefilter('ignore', category=DeprecationWarning)

# The numpy pseudorandom generator seed is specified manually, but you can use
# the random or secrets python library to generate a random seed
SEED = 42


def main():
    """Run the privacy evaluation under an attribute inference adversary."""
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

    # ============================== EVALUATION ===============================
    # Format: {target id => { sensitive attribute => { privacy mechanism => {
    #              iteration number => {
    #                  'AttackerGuess' => [ the guess of the attacker on the
    #                                       value of the sensible attribute ],
    #                  'ProbCorrect' => [ the probability that the attacker
    #                                     make the correct guess ],
    #                  'TargetPresence' => [ whether the target is present ]
    #          }}}}}
    results_target_privacy = {
        tid: {
            sens_attr: {
                privacy_model.__name__: {}
                for privacy_model in gen_model_list + san_tech_list}
            for sens_attr in runconfig['sensitiveAttributes']}
        for tid in target_ids}

    # Add the entry for the raw dataset (the baseline)
    for tid in target_ids:
        for sens_attr in runconfig['sensitiveAttributes']:
            results_target_privacy[tid][sens_attr]['Raw'] = {}

    logger.info('\n---- Start the game ----')
    for n_iter in range(runconfig['nIter']):
        logger.info(f'\n--- Game iteration {n_iter + 1} ---')

        # Draw a raw dataset that does not contain the targets
        r_idx = choice(list(raw_pop_without_targets.index),
                       size=runconfig['sizeRawT'], replace=False).tolist()
        raw_t_out = raw_pop_without_targets.loc[r_idx]

        # ============================== ATTACKS ==============================
        # Initialize the attacks
        # Format: {sensitive attribute => initialized attack}
        attacks = {}
        for sens_attr, attr_type in runconfig['sensitiveAttributes'].items():
            if attr_type == 'LinReg':
                attacks[sens_attr] = LinRegAttack(sensitiveAttribute=sens_attr,
                                                  metadata=metadata)
            elif attr_type == 'Classification':
                attacks[sens_attr] = RandForestAttack(
                    sensitiveAttribute=sens_attr, metadata=metadata)

        # Assess advantage raw (i.e., the each of the attacker when given the
        # knowledge of the raw dataset)
        for sens_attr, attack in attacks.items():

            # Train the classifier to infer the value of the sensitive
            # attribute on the raw dataset without the target
            attack.train(raw_t_out)

            # For each target, infer the value of the sensitive attributes
            for tid in target_ids:
                target = targets.loc[[tid]]
                target_aux = target.loc[[tid], attack.knownAttributes]
                target_secret = target.loc[tid, attack.sensitiveAttribute]

                guess = attack.attack(target_aux, attemptLinkage=True,
                                      data=raw_t_out)
                prob_correct = attack.get_likelihood(
                    target_aux, target_secret, attemptLinkage=True,
                    data=raw_t_out)

                results_target_privacy[tid][sens_attr]['Raw'][n_iter] = {
                    'AttackerGuess': [guess],
                    'ProbCorrect': [prob_correct],
                    'TargetPresence': [LABEL_OUT]}

        # For each target, infer the value of the sensitive attribute when the
        # raw dataset additionally contains the target
        for tid in target_ids:
            target = targets.loc[[tid]]
            raw_t_in = raw_t_out.append(target)

            for sens_attr, attack in attacks.items():
                target_aux = target.loc[[tid], attack.knownAttributes]
                target_secret = target.loc[tid, attack.sensitiveAttribute]

                guess = attack.attack(target_aux, attemptLinkage=True,
                                      data=raw_t_in)
                prob_correct = attack.get_likelihood(
                    target_aux, target_secret, attemptLinkage=True,
                    data=raw_t_in)

                results_target_privacy[tid][sens_attr]['Raw'][n_iter][
                    'AttackerGuess'].append(guess)
                results_target_privacy[tid][sens_attr]['Raw'][n_iter][
                    'ProbCorrect'].append(prob_correct)
                results_target_privacy[tid][sens_attr]['Raw'][n_iter][
                    'TargetPresence'].append(LABEL_IN)

        # Assess advantage Syn (i.e., the reach of the attacker when given only
        # access to the synthetic dataset instead of the raw dataset)
        for gen_model in gen_model_list:
            logger.info(f'Start: Evaluation for model {gen_model.__name__}...')

            # Train the generative model on the raw dataset and generate nSynT
            # synthetic datasets
            gen_model.fit(raw_t_out)
            syn_without_target = [
                gen_model.generate_samples(runconfig['sizeSynT'])
                for _ in range(runconfig['nSynT'])]

            # For each attack on the sensitive attributes
            for sens_attr, attack in attacks.items():
                # Prepare the storage of the result for each target
                for tid in target_ids:
                    results_target_privacy[tid][sens_attr][gen_model.__name__][
                        n_iter] = {
                            'AttackerGuess': [],
                            'ProbCorrect': [],
                            'TargetPresence': [
                                LABEL_OUT for _ in range(runconfig['nSynT'])]}

                # For each synthetic dataset, infer the sensitive attribute
                for syn in syn_without_target:
                    attack.train(syn)

                    for tid in target_ids:
                        target = targets.loc[[tid]]
                        target_aux = target.loc[[tid], attack.knownAttributes]
                        target_secret = target.loc[
                            tid, attack.sensitiveAttribute]

                        guess = attack.attack(target_aux)
                        prob_correct = attack.get_likelihood(
                            target_aux, target_secret)

                        results_target_privacy[tid][sens_attr][
                            gen_model.__name__][n_iter][
                                'AttackerGuess'].append(guess)
                        results_target_privacy[tid][sens_attr][
                            gen_model.__name__][n_iter][
                                'ProbCorrect'].append(prob_correct)

            del syn_without_target

            # For each target, evaluate the model when the target is in the raw
            # dataset used to train the generative model
            for tid in target_ids:
                logger.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                raw_t_in = raw_t_out.append(target)

                # Train the generative model on the raw dataset with the target
                # and generate nSynT synthetic datasets
                gen_model.fit(raw_t_in)
                syn_t_with_target = [
                    gen_model.generate_samples(runconfig['sizeSynT'])
                    for _ in range(runconfig['nSynT'])]

                # For each attack on the sensitive attributes
                for sens_attr, attack in attacks.items():
                    target_aux = target.loc[[tid], attack.knownAttributes]
                    target_secret = target.loc[tid, attack.sensitiveAttribute]

                    # For each synthetic dataset, infer the sensitive attribute
                    for syn in syn_t_with_target:
                        attack.train(syn)

                        guess = attack.attack(target_aux)
                        prob_correct = attack.get_likelihood(
                            target_aux, target_secret)

                        results_target_privacy[tid][sens_attr][
                            gen_model.__name__][n_iter][
                                'AttackerGuess'].append(guess)
                        results_target_privacy[tid][sens_attr][
                            gen_model.__name__][n_iter][
                                'ProbCorrect'].append(prob_correct)
                        results_target_privacy[tid][sens_attr][
                            gen_model.__name__][n_iter][
                                'TargetPresence'].append(LABEL_IN)

            del syn_t_with_target

        # Evaluate each sanitization technique
        for san in san_tech_list:
            logger.info(f'Start: Evaluation for sanitiser {san.__name__}...')

            # Initialize the attacks
            # Format: {sensitive attribute => initialized attack}
            attacks = {}
            for sens_attr, attr_type in runconfig['sensitiveAttributes'
                                                  ].items():
                if attr_type == 'LinReg':
                    attacks[sens_attr] = LinRegAttack(
                        sensitiveAttribute=sens_attr, metadata=metadata,
                        quids=san.quids)
                elif attr_type == 'Classification':
                    attacks[sens_attr] = RandForestAttack(
                        sensitiveAttribute=sens_attr, metadata=metadata,
                        quids=san.quids)

            # Generate the sanitized dataset from the raw dataset without the
            # targets
            san_out = san.sanitise(raw_t_out)

            # For each attack on the sensitive attributes
            for sens_attr, attack in attacks.items():
                attack.train(san_out)

                for tid in target_ids:
                    target = targets.loc[[tid]]
                    target_aux = target.loc[[tid], attack.knownAttributes]
                    target_secret = target.loc[tid, attack.sensitiveAttribute]

                    guess = attack.attack(target_aux, attemptLinkage=True,
                                          data=san_out)
                    prob_correct = attack.get_likelihood(
                        target_aux, target_secret, attemptLinkage=True,
                        data=san_out)

                    results_target_privacy[tid][sens_attr][san.__name__][
                        n_iter] = {
                            'AttackerGuess': [guess],
                            'ProbCorrect': [prob_correct],
                            'TargetPresence': [LABEL_OUT]}

            # Repeat the process with the target in the raw dataset
            for tid in target_ids:
                logger.info(f'Target: {tid}')
                target = targets.loc[[tid]]
                raw_t_in = raw_t_out.append(target)
                san_in = san.sanitise(raw_t_in)

                for sens_attr, attack in attacks.items():
                    target_aux = target.loc[[tid], attack.knownAttributes]
                    target_secret = target.loc[tid, attack.sensitiveAttribute]

                    attack.train(san_in)

                    guess = attack.attack(target_aux, attemptLinkage=True,
                                          data=san_in)
                    prob_correct = attack.get_likelihood(
                        target_aux, target_secret, attemptLinkage=True,
                        data=san_in)

                    results_target_privacy[tid][sens_attr][san.__name__][
                        n_iter]['AttackerGuess'].append(guess)
                    results_target_privacy[tid][sens_attr][san.__name__][
                        n_iter]['ProbCorrect'].append(prob_correct)
                    results_target_privacy[tid][sens_attr][san.__name__][
                        n_iter]['TargetPresence'].append(LABEL_IN)

    # Write the results in the json output file
    output_path = Path(args.outdir) / Path(f'ResultsMLEAI_{data_name}.json')
    logger.info(f'Write results to {output_path}')
    with open(output_path, 'w+') as output_file:
        json.dump(results_target_privacy, output_file, indent=2,
                  default=json_numpy_serialzer)


if __name__ == "__main__":
    main()
