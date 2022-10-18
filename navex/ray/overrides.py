import json
import logging
import os
import sys
import time

import ray
from ray._private.usage import usage_lib
import ray._private.services as services
import ray.ray_constants as ray_constants
import ray._private.utils

from ray.autoscaler._private.fake_multi_node.node_provider import FAKE_HEAD_NODE_ID
from ray.autoscaler._private.cli_logger import cli_logger, cf

logger = logging.getLogger(__name__)

# monkey patching node ip address so that can use tunneling
services.get_node_ip_address = lambda x=None: '127.0.0.1'
services.resolve_ip_for_localhost = lambda x: x

# injecting a sleep as might help with dying / hanging trials, see https://github.com/ray-project/ray/issues/11239
# from ray.tune.trial_runner import TrialRunner
# _parent_get_next_trial = TrialRunner._get_next_trial
# def _my_get_next_trial(self):
#     val = _parent_get_next_trial(self)
#     time.sleep(2)
#     return val
# TrialRunner._get_next_trial = _my_get_next_trial
#
# _parent_process_trial_failure = TrialRunner._process_trial_failure
# def _my_process_trial_failure(self, trial, error_msg):
#     time.sleep(120)
#     _parent_process_trial_failure(self, trial, error_msg)
# TrialRunner._process_trial_failure = _my_process_trial_failure


# from ..\site-packages\ray\scripts\scripts.py
#       - i.e. from https://github.com/ray-project/ray/blob/ray-1.13.0/python/ray/scripts/scripts.py
#       - only the arguments (adding the defaults) and the last two rows of the function have been modified (!)
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
    dashboard_agent_listen_port=0,
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

    try:
        resources = json.loads(resources)
    except Exception:
        cli_logger.error("`{}` is not a valid JSON string.", cf.bold("--resources"))
        cli_logger.abort(
            "Valid values look like this: `{}`",
            cf.bold('--resources=\'{"CustomResource3": 1, ' '"CustomResource2": 2}\''),
        )

        raise Exception(
            "Unable to parse the --resources argument using "
            "json.loads. Try using a format like\n\n"
            '    --resources=\'{"CustomResource1": 3, '
            '"CustomReseource2": 2}\''
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

    if head:
        # Start head node.

        if disable_usage_stats:
            usage_lib.set_usage_stats_enabled_via_env_var(False)
        usage_lib.show_usage_stats_prompt()
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
            cli_logger.print(
                "Will use `{}` as external Redis server address(es). "
                "If the primary one is not reachable, we starts new one(s) "
                "with `{}` in local.",
                cf.bold(address),
                cf.bold("--port"),
            )
            external_addresses = address.split(",")

            # We reuse primary redis as sharding when there's only one
            # instance provided.
            if len(external_addresses) == 1:
                external_addresses.append(external_addresses[0])
            reachable = False
            try:
                [primary_redis_ip, port] = external_addresses[0].split(":")
                ray._private.services.wait_for_redis_to_start(
                    primary_redis_ip, port, password=redis_password
                )
                reachable = True
            # We catch a generic Exception here in case someone later changes
            # the type of the exception.
            except Exception:
                cli_logger.print(
                    "The primary external redis server `{}` is not reachable. "
                    "Will starts new one(s) with `{}` in local.",
                    cf.bold(external_addresses[0]),
                    cf.bold("--port"),
                )
            if reachable:
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
            bootstrap_addresses = services.find_bootstrap_address()
            if default_address in bootstrap_addresses:
                raise ConnectionError(
                    f"Ray is trying to start at {default_address}, "
                    f"but is already running at {bootstrap_addresses}. "
                    "Please specify a different port using the `--port`"
                    " flag of `ray start` command."
                )

        node = ray.node.Node(
            ray_params, head=True, shutdown_at_exit=block, spawn_reaper=block
        )

        bootstrap_addresses = node.address
        if temp_dir is None:
            # Default temp directory.
            temp_dir = ray._private.utils.get_user_temp_dir()
        # Using the user-supplied temp dir unblocks on-prem
        # users who can't write to the default temp.
        current_cluster_path = os.path.join(temp_dir, "ray_current_cluster")
        # TODO: Consider using the custom temp_dir for this file across the
        # code base. (https://github.com/ray-project/ray/issues/16458)
        with open(current_cluster_path, "w") as f:
            print(bootstrap_addresses, file=f)

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
                bootstrap_addresses,
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
        bootstrap_address = services.canonicalize_bootstrap_address(address)

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

        node = ray.node.Node(
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
                "This command will now block until terminated by a signal."
            )
            cli_logger.print(
                "Running subprocesses are monitored and a message will be "
                "printed if any of them terminate unexpectedly."
            )
            cli_logger.flush()

        while True:
            time.sleep(1)
            deceased = node.dead_processes()
            if len(deceased) > 0:
                cli_logger.newline()
                cli_logger.error("Some Ray subprcesses exited unexpectedly:")

                with cli_logger.indented():
                    for process_type, process in deceased:
                        cli_logger.error(
                            "{}",
                            cf.bold(str(process_type)),
                            _tags={"exit code": str(process.returncode)},
                        )

                # shutdown_at_exit will handle cleanup.
                cli_logger.newline()
                cli_logger.error("Remaining processes will be killed.")
                sys.exit(1)
    else:
        return node  # EDITED: own addition
