""""
File to run experiments.
"""
import argparse
from collections import OrderedDict

from experiments.experiments_acic import AcicLinearGenerator, AcicOrigGenerator
from experiments.experiments_ihdp import IHDPDataGenerator
from experiments.main_experiment import get_criterion, get_learner, get_model, run_repeat_experiment

DEFAULT_REGRESSORS = ["lr", "xgb_cv"]
DEFAULT_CRITERIA = ["factual", "oracle", "s", "s_ext", "t", "w_factual", "dr_plug", "r_plug", "dr", "r"]
DEFAULT_LEARNERS = ["s", "s_ext", "t", "dr", "r"]
DEFAULT_PARAM_LOOP = {"sp_1": [0, 0.1, 0.3]}
DEFAULT_ACIC_SIMUS = [2, 7, 26]


def grid_comparison_experiment(
    learner_models=None,
    crit_models=None,
    learner_names=None,
    criterion_names=None,
    experiment_type="acic-simu",
    param_loop=None,
    file_suffix=None,
    n_exp=5,
    n_train=1000,
    n_val=500,
    sp_lin=0.3,
    sp_nonlin=0.2,
    propensity_type="prog",
    p_0=0,
    inter=3,
    inter_t=0,
    sp_1=0,
    xi=0.5,
    nonlin_scale=1,
    error_sd=0.1,
    subset_cols="missp_out",
    trans_cov=0,
    include_oracles=True,
    n_folds=1,
    rel_prop=0.5,
    prop_truespec=True,
):
    if learner_models is None:
        learner_models = DEFAULT_REGRESSORS
    if crit_models is None:
        crit_models = DEFAULT_REGRESSORS
    if learner_names is None:
        learner_names = DEFAULT_LEARNERS
    if criterion_names is None:
        criterion_names = DEFAULT_CRITERIA
    if param_loop is None:
        param_loop = DEFAULT_PARAM_LOOP

    file_name = "results" if file_suffix is None else "results_" + file_suffix

    learners_to_test = {}
    for reg_name in learner_models:
        for learner_name in learner_names:
            learners_to_test.update(
                {learner_name + "-" + reg_name: get_learner(learner_name, get_model(reg_name), n_folds=n_folds)}
            )

    learners_to_test = OrderedDict(learners_to_test)

    if include_oracles:
        oracle_baselines = {}
        for reg_name in learner_models:
            oracle_baselines.update({"oracle-" + reg_name: get_model(reg_name)})
    else:
        oracle_baselines = None

    criteria_to_test = {}
    for crit in criterion_names:
        if crit in ["oracle", "factual", "po_oracle", "w_factual", "w_factual_prefit", "match", "pw"]:
            # criteria that need no regressors
            criteria_to_test.update({crit: get_criterion(crit, None)})
        else:
            for reg in crit_models:
                criteria_to_test.update({crit + "-" + reg: get_criterion(crit, get_model(reg))})

    criteria_to_test = OrderedDict(criteria_to_test)

    if experiment_type == "acic-simu":
        acic_simu_params = {
            "sp_nonlin": sp_nonlin,
            "sp_lin": sp_lin,
            "sp_1": sp_1,
            "p_0": p_0,
            "inter": inter,
            "inter_t": inter_t,
            "n_train": n_train,
            "n_val": n_val,
            "subset_cols": subset_cols,
            "xi": xi,
            "propensity_type": propensity_type,
            "trans_cov": trans_cov,
            "nonlin_scale": nonlin_scale,
            "error_sd": error_sd,
            "rel_prop": rel_prop,
            "prop_truespec": prop_truespec,
        }

        param_name = next(iter(param_loop))
        for param in param_loop[param_name]:
            acic_simu_params.update({param_name: param})

            run_repeat_experiment(
                learners_to_test,
                criteria_to_compare=criteria_to_test,
                dgp_to_sample=AcicLinearGenerator(**acic_simu_params),
                file_name=file_name,
                n_exp=n_exp,
                crit_id="mixed" if len(crit_models) > 1 else crit_models[0],
                learner_id="mixed" if len(learner_models) > 1 else learner_models[0],
                oracle_baselines=oracle_baselines,
            )

    elif experiment_type == "acic-orig":
        for simu_num in DEFAULT_ACIC_SIMUS:
            run_repeat_experiment(
                learners_to_test,
                criteria_to_compare=criteria_to_test,
                dgp_to_sample=AcicOrigGenerator(n_train=n_train, n_val=n_val, simu_num=simu_num),
                file_name=file_name,
                crit_id="mixed" if len(crit_models) > 1 else crit_models[0],
                learner_id="mixed" if len(learner_models) > 1 else learner_models[0],
                oracle_baselines=oracle_baselines,
            )

    elif experiment_type == "ihdp":
        run_repeat_experiment(
            learners_to_test,
            criteria_to_compare=criteria_to_test,
            dgp_to_sample=IHDPDataGenerator(setting="original", n_train=n_train),
            file_name=file_name,
            n_exp=n_exp,
            crit_id="mixed" if len(crit_models) > 1 else crit_models[0],
            learner_id="mixed" if len(learner_models) > 1 else learner_models[0],
            oracle_baselines=oracle_baselines,
        )

        run_repeat_experiment(
            learners_to_test,
            criteria_to_compare=criteria_to_test,
            dgp_to_sample=IHDPDataGenerator(setting="modified", n_train=n_train),
            file_name=file_name,
            n_exp=n_exp,
            crit_id="mixed" if len(crit_models) > 1 else crit_models[0],
            learner_id="mixed" if len(learner_models) > 1 else learner_models[0],
            oracle_baselines=oracle_baselines,
        )

    else:
        raise ValueError("Unknown experiment type.")


def init_arg():
    # arg parser if script is run from shell
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_type", default="acic-simu", type=str)
    parser.add_argument("--setup", default="A", type=str)
    parser.add_argument("--regressors", default=DEFAULT_REGRESSORS, type=str, nargs="+")
    parser.add_argument("--learners", default=DEFAULT_LEARNERS, type=str, nargs="+")
    parser.add_argument("--criteria", default=DEFAULT_CRITERIA, type=str, nargs="+")
    parser.add_argument("--file_name", default=None, type=str)
    parser.add_argument("--n_repeats", default=5, type=int, nargs="+")
    parser.add_argument("--n_train", default=1000, type=int)
    parser.add_argument("--n_val", default=500, type=int)
    parser.add_argument("--rel_prop", default=0.5, type=float)
    parser.add_argument("--true_prop", default=True, type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg()
    n_repeats = args.n_repeats if len(args.n_repeats) > 1 else args.n_repeats[0]
    if args.experiment_type == "acic-simu":
        if args.setup == "A":
            grid_comparison_experiment(
                learner_models=args.regressors,
                crit_models=args.regressors,
                learner_names=args.learners,
                criterion_names=args.criteria,
                experiment_type="acic-simu",
                file_suffix=args.file_name,
                n_exp=n_repeats,
                n_train=args.n_train,
                n_val=args.n_val,
                xi=0,
                trans_cov=0,
                rel_prop=args.rel_prop,
                prop_truespec=args.true_prop,
            )
            print("dfd")
        elif args.setup == "B":
            grid_comparison_experiment(
                learner_models=args.regressors,
                crit_models=args.regressors,
                learner_names=args.learners,
                criterion_names=args.criteria,
                experiment_type="acic-simu",
                file_suffix=args.file_name,
                n_exp=n_repeats,
                n_train=args.n_train,
                n_val=args.n_val,
                xi=0,
                trans_cov=1,
                rel_prop=args.rel_prop,
                prop_truespec=args.true_prop,
            )
        elif args.setup == "C":
            grid_comparison_experiment(
                learner_models=args.regressors,
                crit_models=args.regressors,
                learner_names=args.learners,
                criterion_names=args.criteria,
                experiment_type="acic-simu",
                file_suffix=args.file_name,
                n_exp=n_repeats,
                n_train=args.n_train,
                n_val=args.n_val,
                xi=3,
                trans_cov=0,
                rel_prop=args.rel_prop,
                prop_truespec=args.true_prop,
            )
        elif args.setup == "D":
            grid_comparison_experiment(
                learner_models=args.regressors,
                crit_models=args.regressors,
                learner_names=args.learners,
                criterion_names=args.criteria,
                experiment_type="acic-simu",
                file_suffix=args.file_name,
                n_exp=n_repeats,
                n_train=args.n_train,
                n_val=args.n_val,
                xi=3,
                trans_cov=1,
                rel_prop=args.rel_prop,
                prop_truespec=args.true_prop,
            )
    elif args.experiment_type == "acic-orig":
        grid_comparison_experiment(
            learner_models=args.regressors,
            crit_models=args.regressors,
            learner_names=args.learners,
            criterion_names=args.criteria,
            file_suffix=args.file_name,
            experiment_type="acic-orig",
            subset_cols="all",
            n_exp=n_repeats,
            n_train=args.n_train,
            n_val=args.n_val,
        )
    elif args.experiment_type == "ihdp":
        grid_comparison_experiment(
            learner_models=args.regressors,
            crit_models=args.regressors,
            learner_names=args.learners,
            criterion_names=args.criteria,
            experiment_type="ihdp",
            n_exp=n_repeats,
            n_train=0.66,  # pyright: ignore
            file_suffix=args.file_name,
        )
    else:
        raise ValueError("Unknown experiment type.")
