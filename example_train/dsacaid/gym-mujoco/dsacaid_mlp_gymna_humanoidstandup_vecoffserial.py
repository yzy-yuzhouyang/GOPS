import argparse

from gops.create_pkg.create_alg import create_alg
from gops.create_pkg.create_buffer import create_buffer
from gops.create_pkg.create_env import create_env
from gops.create_pkg.create_evaluator import create_evaluator
from gops.create_pkg.create_sampler import create_sampler
from gops.create_pkg.create_trainer import create_trainer
from gops.utils.init_args import init_args
from gops.utils.plot_evaluation import plot_all
from gops.utils.tensorboard_setup import start_tensorboard, save_tb_to_csv


if __name__ == "__main__":
    # Parameters Setup
    parser = argparse.ArgumentParser()

    # =========================================================================
    # Key Parameters for users
    # =========================================================================
    parser.add_argument("--env_id", type=str, default="gymna_humanoidstandup", help="Id of environment")
    parser.add_argument("--algorithm", type=str, default="DSACAID", help="RL algorithm")
    parser.add_argument("--enable_cuda", type=bool, default=True, help="Enable CUDA")
    parser.add_argument("--seed", type=int, default=12345, help="Global seed")
    parser.add_argument("--exp_tag", type=str, default="", help="Experiment tag for logging and identification")
    # =========================================================================
    # AID Mechanism Hyperparameters
    # =========================================================================
    # 1. Pessimistic Evaluation (Critic)
    parser.add_argument("--num_q", type=int, default=4, help="Ensemble size")
    # 1.1 Pessimistic target
    parser.add_argument("--lambda_lower", type=float, default=0.5, help="Lower confidence bound coefficient")
    parser.add_argument("--beta", type=float, default=0, help="Initial aleatoric pessimism coefficient")
    parser.add_argument("--beta_annealing_rate", type=float, default=2e-3, help="Annealing rate for beta")
    parser.add_argument("--freeze_early_beta", type=bool, default=False, help="Whether to freeze beta decay during the initial warmup phase")
    parser.add_argument("--q_bias_lower_threshold", type=float, default=-1000, help="Bias threshold to trigger or resume beta updates")
    # 1.2 Gradient modulation
    parser.add_argument("--use_homogeneous_sigma_step_ratio", type=bool, default=True, help="Use homogeneous modulation ratio for sigma")
    # 2. Optimistic exploration (Actor)
    parser.add_argument("--lambda_upper", type=float, default=0.5, help="Upper confidence bound coefficient")
    # 3. Legacy (Not used in formal experiments)
    parser.add_argument("--enable_epi_step_scale", type=bool, default=True, help="(Unused) Involve epistemic uncertainty in gradient modulation")
    parser.add_argument("--entropy_scale_ratio", type=float, default=1.0, help="(Unused) Scaling factor for target entropy")

    ################################################
    # 1. Parameters for environment
    parser.add_argument("--vector_env_num", type=int, default=4, help="Number of vector envs")
    parser.add_argument("--vector_env_type", type=str, default='async', help="Options: sync/async")
    parser.add_argument("--gym2gymnasium", type=bool, default=True, help="Convert Gym-style env to Gymsnaium-style")
    parser.add_argument("--reward_scale", type=float, default=0.1, help="reward scale factor")
    parser.add_argument("--is_render", type=bool, default=False, help="Draw environment animation")
    parser.add_argument("--is_adversary", type=bool, default=False, help="Adversary training")

    ################################################
    # 2.1 Parameters of value approximate function
    parser.add_argument(
        "--value_func_name",
        type=str,
        default="RegularizedActionValueDistri",
        help="RegularizedActionValueDistri/ActionValueDistri",
    )
    parser.add_argument("--value_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS")
    value_func_type = parser.parse_known_args()[0].value_func_type
    parser.add_argument("--value_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--value_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument("--value_output_activation", type=str, default="linear", help="Options: linear/tanh")

    # 2.2 Parameters of policy approximate function
    parser.add_argument(
        "--policy_func_name",
        type=str,
        default="RegularizedStochaPolicy",
        help="Options: RegularizedStochaPolicy/StochaPolicy",
    )
    parser.add_argument(
        "--policy_func_type", type=str, default="MLP", help="Options: MLP/CNN/CNN_SHARED/RNN/POLY/GAUSS"
    )
    parser.add_argument(
        "--policy_act_distribution",
        type=str,
        default="TanhGaussDistribution",
        help="Options: default/TanhGaussDistribution/GaussDistribution",
    )
    policy_func_type = parser.parse_known_args()[0].policy_func_type
    parser.add_argument("--policy_hidden_sizes", type=list, default=[256,256,256])
    parser.add_argument(
        "--policy_hidden_activation", type=str, default="relu", help="Options: relu/gelu/elu/selu/sigmoid/tanh"
    )
    parser.add_argument(
        "--policy_output_activation", type=str, default="linear", help="Options: linear/tanh"
    )
    parser.add_argument("--policy_min_log_std", type=int, default=-20)
    parser.add_argument("--policy_max_log_std", type=int, default=0.5)

    ################################################
    # 3. Parameters for RL algorithm
    parser.add_argument("--value_learning_rate", type=float, default=0.0003)
    parser.add_argument("--policy_learning_rate", type=float, default=0.0003)
    parser.add_argument("--alpha_learning_rate", type=float, default=0.0003)
    parser.add_argument("--use_huber_loss", type=bool, default=False, help="Use huber loss")
    parser.add_argument("--q_delta_in_huber_loss", type=float, default=50.0, help="Delta parameter for q loss")
    parser.add_argument("--sigma_delta_in_sigma_loss", type=float, default=50.0, help="Delta parameter for sigma loss")
    # special parameter
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=bool, default=True)
    parser.add_argument("--delay_update", type=int, default=2)

    ################################################
    # 4. Parameters for trainer
    parser.add_argument(
        "--trainer",
        type=str,
        default="off_serial_dsacaid_trainer",
        help="Options: on_serial_trainer, on_sync_trainer, off_serial_trainer, off_async_trainer",
    )
    # Maximum iteration number
    parser.add_argument("--max_iteration", type=int, default=1500000)
    parser.add_argument(
        "--ini_network_dir",
        type=str,
        default=None
    )
    trainer_type = parser.parse_known_args()[0].trainer

    # 4.1. Parameters for off_serial_trainer
    parser.add_argument(
        "--buffer_name", type=str, default="replay_buffer", help="Options:replay_buffer/prioritized_replay_buffer"
    )
    # Size of collected samples before training
    parser.add_argument("--buffer_warm_size", type=int, default=10000)
    # Max size of reply buffer
    parser.add_argument("--buffer_max_size", type=int, default=2*500000)
    # Batch size of replay samples from buffer
    parser.add_argument("--replay_batch_size", type=int, default=256)
    # Period of sampling
    parser.add_argument("--sample_interval", type=int, default=1)

    ################################################
    # 5. Parameters for sampler
    parser.add_argument("--sampler_name", type=str, default="off_dsacaid_sampler", help="Options: on_sampler/off_sampler")
    # Batch size of sampler for buffer store
    parser.add_argument("--sample_batch_size", type=int, default=20)
    # Add noise to action for better exploration
    parser.add_argument("--noise_params", type=dict, default=None)

    ################################################
    # 6. Parameters for evaluator
    parser.add_argument("--evaluator_name", type=str, default="evaluator_dsacaid")
    parser.add_argument("--num_eval_episode", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=2500)
    parser.add_argument("--eval_save", type=str, default=False, help="save evaluation data")

    ################################################
    # 7. Data savings
    parser.add_argument("--save_folder", type=str, default= None)
    # Save value/policy every N updates
    parser.add_argument("--apprfunc_save_interval", type=int, default=50000)
    # Save key info every N updates
    parser.add_argument("--log_save_interval", type=int, default=10000)

    ################################################
    # Get parameter dictionary
    args = vars(parser.parse_args())
    env = create_env(**{**args, "vector_env_num": None})
    args = init_args(env, **args)

    start_tensorboard(args["save_folder"])
    # Step 1: create algorithm and approximate function
    alg = create_alg(**args)
    # Step 2: create sampler in trainer
    sampler = create_sampler(**args)
    # Step 3: create buffer in trainer
    buffer = create_buffer(**args)
    # Step 4: create evaluator in trainer
    evaluator = create_evaluator(**args)
    # Step 5: create trainer
    trainer = create_trainer(alg, sampler, buffer, evaluator, **args)

    ################################################
    # Start training ... ...
    trainer.train()
    print("Training is finished!")

    ################################################
    # Plot and save training figures
    plot_all(args["save_folder"])
    save_tb_to_csv(args["save_folder"])
    print("Plot & Save are finished!")
