import logging
import os
import signal
import time
import warnings

import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._private.utils import parse_resources_json, get_ray_address_file
from ray._private.storage import _load_class
from ray._private.usage import usage_lib
from ray.autoscaler._private.cli_logger import cf, cli_logger

from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID

logger = logging.getLogger(__name__)

# monkey patching node ip address so that can use tunneling
# NOTE: might not work, consider editing source directly <env-path>/lib/python3.8/site-packages/ray/_private/services.py
services.node_ip_address_from_perspective = lambda x: '127.0.0.1'
services.get_node_ip_address = lambda x=None: '127.0.0.1'

# injecting a sleep as might help with dying / hanging trials, see https://github.com/ray-project/ray/issues/11239
# from ray.tune.trial_runner import TrialRunner
# _parent_update_trial_queue = TrialRunner._update_trial_queue
# def _my_update_trial_queue(self, blocking: bool = False, timeout: int = 600):
#     val = _parent_update_trial_queue(self, blocking, timeout)
#     time.sleep(2)
#     return val
# TrialRunner._update_trial_queue = _my_update_trial_queue
#
# _parent_process_trial_failure = TrialRunner._process_trial_failure
# def _my_process_trial_failure(self, trial, exc=None):
#     time.sleep(120)
#     _parent_process_trial_failure(self, trial, exc)
# TrialRunner._process_trial_failure = _my_process_trial_failure


# from ..\site-packages\ray\scripts\scripts.py
#       - i.e. from https://github.com/ray-project/ray/blob/ray-2.2.0/python/ray/scripts/scripts.py
#       - only the arguments (adding the defaults) and the last couple of rows of the function have been modified (!)
#
def start(
    node_ip_address=None,
    address=None,
    port=None,
    node_name=None,
    redis_password=ray_constants.REDIS_DEFAULT_PASSWORD,
    redis_shard_ports=None,
    object_manager_port=None,
    node_manager_port=0,
    gcs_server_port=None,
    min_worker_port=10002,
    max_worker_port=19999,
    worker_port_list=None,
    ray_client_server_port=10001,
    memory=None,
    object_store_memory=None,
    redis_max_memory=None,
    num_cpus=None,
    num_gpus=None,
    resources="{}",
    head=False,
    include_dashboard=None,
    dashboard_host="localhost",
    dashboard_port=ray_constants.DEFAULT_DASHBOARD_PORT,
    dashboard_agent_listen_port=ray_constants.DEFAULT_DASHBOARD_AGENT_LISTEN_PORT,
    dashboard_agent_grpc_port=None,
    block=False,
    plasma_directory=None,
    autoscaling_config=None,
    no_redirect_output=False,
    plasma_store_socket_name=None,
    raylet_socket_name=None,
    temp_dir=None,
    storage=None,
    system_config=None,
    enable_object_reconstruction=False,
    metrics_export_port=None,
    no_monitor=False,
    tracing_startup_hook=None,
    ray_debugger_external=False,
    disable_usage_stats=False,
):
    """Start Ray processes manually on the local machine."""

    if gcs_server_port is not None:
        cli_logger.error(
            "`{}` is deprecated and ignored. Use {} to specify "
            "GCS server port on head node.",
            cf.bold("--gcs-server-port"),
            cf.bold("--port"),
        )

    # Whether the original arguments include node_ip_address.
    include_node_ip_address = False
    if node_ip_address is not None:
        include_node_ip_address = True
        node_ip_address = services.resolve_ip_for_localhost(node_ip_address)

    resources = parse_resources_json(resources, cli_logger, cf)

    if plasma_store_socket_name is not None:
        warnings.warn(
            "plasma_store_socket_name is deprecated and will be removed. You are not "
            "supposed to specify this parameter as it's internal.",
            DeprecationWarning,
            stacklevel=2,
        )
    if raylet_socket_name is not None:
        warnings.warn(
            "raylet_socket_name is deprecated and will be removed. You are not "
            "supposed to specify this parameter as it's internal.",
            DeprecationWarning,
            stacklevel=2,
        )

    redirect_output = None if not no_redirect_output else True
    ray_params = ray._private.parameter.RayParams(
        node_ip_address=node_ip_address,
        node_name=node_name if node_name else node_ip_address,
        min_worker_port=min_worker_port,
        max_worker_port=max_worker_port,
        worker_port_list=worker_port_list,
        ray_client_server_port=ray_client_server_port,
        object_manager_port=object_manager_port,
        node_manager_port=node_manager_port,
        memory=memory,
        object_store_memory=object_store_memory,
        redis_password=redis_password,
        redirect_output=redirect_output,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        resources=resources,
        autoscaling_config=autoscaling_config,
        plasma_directory=plasma_directory,
        huge_pages=False,
        plasma_store_socket_name=plasma_store_socket_name,
        raylet_socket_name=raylet_socket_name,
        temp_dir=temp_dir,
        storage=storage,
        include_dashboard=include_dashboard,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        dashboard_agent_listen_port=dashboard_agent_listen_port,
        metrics_agent_port=dashboard_agent_grpc_port,
        _system_config=system_config,
        enable_object_reconstruction=enable_object_reconstruction,
        metrics_export_port=metrics_export_port,
        no_monitor=no_monitor,
        tracing_startup_hook=tracing_startup_hook,
        ray_debugger_external=ray_debugger_external,
    )

    if ray_constants.RAY_START_HOOK in os.environ:
        _load_class(os.environ[ray_constants.RAY_START_HOOK])(ray_params, head)

    if head:
        # Start head node.

        if disable_usage_stats:
            usage_lib.set_usage_stats_enabled_via_env_var(False)
        usage_lib.show_usage_stats_prompt(cli=True)
        cli_logger.newline()

        if port is None:
            port = ray_constants.DEFAULT_PORT

        # Set bootstrap port.
        assert ray_params.redis_port is None
        assert ray_params.gcs_server_port is None
        ray_params.gcs_server_port = port

        if os.environ.get("RAY_FAKE_CLUSTER"):
            ray_params.env_vars = {
                "RAY_OVERRIDE_NODE_ID_FOR_TESTING": FAKE_HEAD_NODE_ID
            }

        num_redis_shards = None
        # Start Ray on the head node.
        if redis_shard_ports is not None and address is None:
            redis_shard_ports = redis_shard_ports.split(",")
            # Infer the number of Redis shards from the ports if the number is
            # not provided.
            num_redis_shards = len(redis_shard_ports)

        # This logic is deprecated and will be removed later.
        if address is not None:
            cli_logger.warning(
                "Specifying {} for external Redis address is deprecated. "
                "Please specify environment variable {}={} instead.",
                cf.bold("--address"),
                cf.bold("RAY_REDIS_ADDRESS"),
                address,
            )
            external_addresses = address.split(",")

            # We reuse primary redis as sharding when there's only one
            # instance provided.
            if len(external_addresses) == 1:
                external_addresses.append(external_addresses[0])

            ray_params.update_if_absent(external_addresses=external_addresses)
            num_redis_shards = len(external_addresses) - 1
            if redis_password == ray_constants.REDIS_DEFAULT_PASSWORD:
                cli_logger.warning(
                    "`{}` should not be specified as empty string if "
                    "external redis server(s) `{}` points to requires "
                    "password.",
                    cf.bold("--redis-password"),
                    cf.bold("--address"),
                )

        # Get the node IP address if one is not provided.
        ray_params.update_if_absent(node_ip_address=services.get_node_ip_address())
        cli_logger.labeled_value("Local node IP", ray_params.node_ip_address)

        # Initialize Redis settings.
        ray_params.update_if_absent(
            redis_shard_ports=redis_shard_ports,
            redis_max_memory=redis_max_memory,
            num_redis_shards=num_redis_shards,
            redis_max_clients=None,
        )

        # Fail early when starting a new cluster when one is already running
        if address is None:
            default_address = f"{ray_params.node_ip_address}:{port}"
            bootstrap_address = services.find_bootstrap_address(temp_dir)
            if (
                default_address == bootstrap_address
                and bootstrap_address in services.find_gcs_addresses()
            ):
                # The default address is already in use by a local running GCS
                # instance.
                raise ConnectionError(
                    f"Ray is trying to start at {default_address}, "
                    f"but is already running at {bootstrap_address}. "
                    "Please specify a different port using the `--port`"
                    " flag of `ray start` command."
                )

        node = ray._private.node.Node(
            ray_params, head=True, shutdown_at_exit=block, spawn_reaper=block
        )

        bootstrap_address = node.address

        # this is a noop if new-style is not set, so the old logger calls
        # are still in place
        cli_logger.newline()
        startup_msg = "Ray runtime started."
        cli_logger.success("-" * len(startup_msg))
        cli_logger.success(startup_msg)
        cli_logger.success("-" * len(startup_msg))
        cli_logger.newline()
        with cli_logger.group("Next steps"):
            cli_logger.print("To connect to this Ray runtime from another node, run")
            # NOTE(kfstorm): Java driver rely on this line to get the address
            # of the cluster. Please be careful when updating this line.
            cli_logger.print(
                cf.bold("  ray start --address='{}'"),
                bootstrap_address,
            )
            if bootstrap_address.startswith("127.0.0.1:"):
                cli_logger.print(
                    "This Ray runtime only accepts connections from local host."
                )
                cli_logger.print(
                    "To accept connections from remote hosts, "
                    "specify a public ip when starting"
                )
                cli_logger.print(
                    "the head node: ray start --head --node-ip-address=<public-ip>."
                )
            cli_logger.newline()
            cli_logger.print("Alternatively, use the following Python code:")
            with cli_logger.indented():
                cli_logger.print("{} ray", cf.magenta("import"))
                # Note: In the case of joining an existing cluster using
                # `address="auto"`, the _node_ip_address parameter is
                # unnecessary.
                cli_logger.print(
                    "ray{}init(address{}{}{})",
                    cf.magenta("."),
                    cf.magenta("="),
                    cf.yellow("'auto'"),
                    ", _node_ip_address{}{}".format(
                        cf.magenta("="), cf.yellow("'" + node_ip_address + "'")
                    )
                    if include_node_ip_address
                    else "",
                )
            cli_logger.newline()
            cli_logger.print(
                "To connect to this Ray runtime from outside of "
                "the cluster, for example to"
            )
            cli_logger.print(
                "connect to a remote cluster from your laptop "
                "directly, use the following"
            )
            cli_logger.print("Python code:")
            with cli_logger.indented():
                cli_logger.print("{} ray", cf.magenta("import"))
                cli_logger.print(
                    "ray{}init(address{}{})",
                    cf.magenta("."),
                    cf.magenta("="),
                    cf.yellow(
                        "'ray://<head_node_ip_address>:" f"{ray_client_server_port}'"
                    ),
                )
            cli_logger.newline()
            cli_logger.print("To see the status of the cluster, use")
            cli_logger.print("  {}".format(cf.bold("ray status")))
            dashboard_url = node.address_info["webui_url"]
            if dashboard_url:
                cli_logger.print("To monitor and debug Ray, view the dashboard at ")
                cli_logger.print(
                    "  {}".format(
                        cf.bold(dashboard_url),
                    )
                )
            cli_logger.newline()
            cli_logger.print(
                cf.underlined(
                    "If connection fails, check your "
                    "firewall settings and "
                    "network configuration."
                )
            )
            cli_logger.newline()
            cli_logger.print("To terminate the Ray runtime, run")
            cli_logger.print(cf.bold("  ray stop"))
        ray_params.gcs_address = bootstrap_address
    else:
        # Start worker node.

        # Ensure `--address` flag is specified.
        if address is None:
            cli_logger.abort(
                "`{}` is a required flag unless starting a head node with `{}`.",
                cf.bold("--address"),
                cf.bold("--head"),
            )
            raise Exception(
                "`--address` is a required flag unless starting a "
                "head node with `--head`."
            )

        # Raise error if any head-only flag are specified.
        head_only_flags = {
            "--port": port,
            "--redis-shard-ports": redis_shard_ports,
            "--include-dashboard": include_dashboard,
        }
        for flag, val in head_only_flags.items():
            if val is None:
                continue
            cli_logger.abort(
                "`{}` should only be specified when starting head node with `{}`.",
                cf.bold(flag),
                cf.bold("--head"),
            )
            raise ValueError(
                f"{flag} should only be specified when starting head node "
                "with `--head`."
            )

        # Start Ray on a non-head node.
        bootstrap_address = services.canonicalize_bootstrap_address(
            address, temp_dir=temp_dir
        )

        if bootstrap_address is None:
            cli_logger.abort(
                "Cannot canonicalize address `{}={}`.",
                cf.bold("--address"),
                cf.bold(address),
            )
            raise Exception("Cannot canonicalize address " f"`--address={address}`.")

        ray_params.gcs_address = bootstrap_address

        # Get the node IP address if one is not provided.
        ray_params.update_if_absent(
            node_ip_address=services.get_node_ip_address(bootstrap_address)
        )

        cli_logger.labeled_value("Local node IP", ray_params.node_ip_address)

        node = ray._private.node.Node(
            ray_params, head=False, shutdown_at_exit=block, spawn_reaper=block
        )

        # Ray and Python versions should probably be checked before
        # initializing Node.
        node.check_version_info()

        cli_logger.newline()
        startup_msg = "Ray runtime started."
        cli_logger.success("-" * len(startup_msg))
        cli_logger.success(startup_msg)
        cli_logger.success("-" * len(startup_msg))
        cli_logger.newline()
        cli_logger.print("To terminate the Ray runtime, run")
        cli_logger.print(cf.bold("  ray stop"))
        cli_logger.flush()

    if block:
        cli_logger.newline()
        with cli_logger.group(cf.bold("--block")):
            cli_logger.print(
                "This command will now block forever until terminated by a signal."
            )
            cli_logger.print(
                "Running subprocesses are monitored and a message will be "
                "printed if any of them terminate unexpectedly. Subprocesses "
                "exit with SIGTERM will be treated as graceful, thus NOT reported."
            )
            cli_logger.flush()

        while True:
            time.sleep(1)
            deceased = node.dead_processes()

            # Report unexpected exits of subprocesses with unexpected return codes.
            # We are explicitly expecting SIGTERM because this is how `ray stop` sends
            # shutdown signal to subprocesses, i.e. log_monitor, raylet...
            # NOTE(rickyyx): We are treating 128+15 as an expected return code since
            # this is what autoscaler/_private/monitor.py does upon SIGTERM
            # handling.
            expected_return_codes = [
                0,
                signal.SIGTERM,
                -1 * signal.SIGTERM,
                128 + signal.SIGTERM,
            ]
            unexpected_deceased = [
                (process_type, process)
                for process_type, process in deceased
                if process.returncode not in expected_return_codes
            ]
            if len(unexpected_deceased) > 0:
                cli_logger.newline()
                cli_logger.error("Some Ray subprocesses exited unexpectedly:")

                with cli_logger.indented():
                    for process_type, process in unexpected_deceased:
                        cli_logger.error(
                            "{}",
                            cf.bold(str(process_type)),
                            _tags={"exit code": str(process.returncode)},
                        )

                cli_logger.newline()
                cli_logger.error("Remaining processes will be killed.")
                # explicitly kill all processes since atexit handlers
                # will not exit with errors.
                node.kill_all_processes(check_alive=False, allow_graceful=False)
                os._exit(1)
        # not-reachable

    assert ray_params.gcs_address is not None

    os.makedirs(os.path.dirname(get_ray_address_file(temp_dir)), exist_ok=True)     # EDITED: own addition
    ray._private.utils.write_ray_address(ray_params.gcs_address, temp_dir)

    return node   # EDITED: own addition
