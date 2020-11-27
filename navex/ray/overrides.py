import json
import logging
import sys
import time
from socket import socket

import ray
import ray.node
import ray.parameter
import ray._private.services as services
import ray.services as services2

import ray.ray_constants as ray_constants
import ray.utils

from ray.autoscaler._private.cli_logger import cli_logger, cf

from ray.scripts.scripts import check_no_existing_redis_clients

logger = logging.getLogger(__name__)

services.get_node_ip_address = lambda x=None: '127.0.0.1'
services2.get_node_ip_address = lambda x=None: '127.0.0.1'
SKIP_VERSION_CHECK = True


def start(node_ip_address=None, address=None, port=None, redis_password=ray_constants.REDIS_DEFAULT_PASSWORD,
          redis_shard_ports=None, object_manager_port=None, node_manager_port=None, gcs_server_port=None,
          min_worker_port=10000, max_worker_port=10999, worker_port_list=None, memory=None,
          object_store_memory=None, redis_max_memory=None, num_cpus=None, num_gpus=None, resources="{}",
          head=False, include_dashboard=None, dashboard_host="localhost",
          dashboard_port=ray_constants.DEFAULT_DASHBOARD_PORT, block=False,
          plasma_directory=None, autoscaling_config=None, no_redirect_worker_output=False,
          no_redirect_output=False, plasma_store_socket_name=None, raylet_socket_name=None,
          temp_dir=None, java_worker_options=None, load_code_from_local=False,
          code_search_path=None, system_config=None, lru_evict=False,
          enable_object_reconstruction=False, metrics_export_port=None, log_style="auto",
          log_color="auto", verbose=False):
    """Start Ray processes manually on the local machine."""
    cli_logger.configure(log_style, log_color, verbose)
    if gcs_server_port and not head:
        raise ValueError(
            "gcs_server_port can be only assigned when you specify --head.")

    # Convert hostnames to numerical IP address.
    if node_ip_address is not None:
        node_ip_address = services.address_to_ip(node_ip_address)

    redis_address = None
    if address is not None:
        (redis_address, redis_address_ip,
         redis_address_port) = services.validate_redis_address(address)
    try:
        resources = json.loads(resources)
    except Exception:
        cli_logger.error("`{}` is not a valid JSON string.",
                         cf.bold("--resources"))
        cli_logger.abort(
            "Valid values look like this: `{}`",
            cf.bold("--resources='\"CustomResource3\": 1, "
                    "\"CustomResource2\": 2}'"))

        raise Exception("Unable to parse the --resources argument using "
                        "json.loads. Try using a format like\n\n"
                        "    --resources='{\"CustomResource1\": 3, "
                        "\"CustomReseource2\": 2}'")

    redirect_worker_output = None if not no_redirect_worker_output else True
    redirect_output = None if not no_redirect_output else True
    ray_params = ray.parameter.RayParams(
        node_ip_address=node_ip_address,
        min_worker_port=min_worker_port,
        max_worker_port=max_worker_port,
        worker_port_list=worker_port_list,
        object_manager_port=object_manager_port,
        node_manager_port=node_manager_port,
        gcs_server_port=gcs_server_port,
        memory=memory,
        object_store_memory=object_store_memory,
        redis_password=redis_password,
        redirect_worker_output=redirect_worker_output,
        redirect_output=redirect_output,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        resources=resources,
        plasma_directory=plasma_directory,
        huge_pages=False,
        plasma_store_socket_name=plasma_store_socket_name,
        raylet_socket_name=raylet_socket_name,
        temp_dir=temp_dir,
        include_dashboard=include_dashboard,
        dashboard_host=dashboard_host,
        dashboard_port=dashboard_port,
        java_worker_options=java_worker_options,
        load_code_from_local=load_code_from_local,
        code_search_path=code_search_path,
        _system_config=system_config,
        lru_evict=lru_evict,
        enable_object_reconstruction=enable_object_reconstruction,
        metrics_export_port=metrics_export_port)
    if head:
        # Use default if port is none, allocate an available port if port is 0
        if port is None:
            port = ray_constants.DEFAULT_PORT

        if port == 0:
            with socket() as s:
                s.bind(("", 0))
                port = s.getsockname()[1]

        num_redis_shards = None
        # Start Ray on the head node.
        if redis_shard_ports is not None:
            redis_shard_ports = redis_shard_ports.split(",")
            # Infer the number of Redis shards from the ports if the number is
            # not provided.
            num_redis_shards = len(redis_shard_ports)

        if redis_address is not None:
            cli_logger.abort(
                "`{}` starts a new Redis server, `{}` should not be set.",
                cf.bold("--head"), cf.bold("--address"))

            raise Exception("If --head is passed in, a Redis server will be "
                            "started, so a Redis address should not be "
                            "provided.")

        node_ip_address = services.get_node_ip_address()

        # Get the node IP address if one is not provided.
        ray_params.update_if_absent(node_ip_address=node_ip_address)
        cli_logger.labeled_value("Local node IP", ray_params.node_ip_address)
        ray_params.update_if_absent(
            redis_port=port,
            redis_shard_ports=redis_shard_ports,
            redis_max_memory=redis_max_memory,
            num_redis_shards=num_redis_shards,
            redis_max_clients=None,
            autoscaling_config=autoscaling_config,
        )

        # Fail early when starting a new cluster when one is already running
        if address is None:
            default_address = f"{node_ip_address}:{port}"
            redis_addresses = services.find_redis_address(default_address)
            if len(redis_addresses) > 0:
                raise ConnectionError(
                    f"Ray is already running at {default_address}. "
                    f"Please specify a different port using the `--port`"
                    f" command to `ray start`.")

        node = ray.node.Node(
            ray_params, head=True, shutdown_at_exit=block, spawn_reaper=block)
        redis_address = node.redis_address

        # this is a noop if new-style is not set, so the old logger calls
        # are still in place
        cli_logger.newline()
        startup_msg = "Ray runtime started."
        cli_logger.success("-" * len(startup_msg))
        cli_logger.success(startup_msg)
        cli_logger.success("-" * len(startup_msg))
        cli_logger.newline()
        with cli_logger.group("Next steps"):
            cli_logger.print(
                "To connect to this Ray runtime from another node, run")
            cli_logger.print(
                cf.bold("  ray start --address='{}'{}"), redis_address,
                f" --redis-password='{redis_password}'"
                if redis_password else "")
            cli_logger.newline()
            cli_logger.print("Alternatively, use the following Python code:")
            with cli_logger.indented():
                with cf.with_style("monokai") as c:
                    cli_logger.print("{} ray", c.magenta("import"))
                    cli_logger.print(
                        "ray{}init(address{}{}{})", c.magenta("."),
                        c.magenta("="), c.yellow("'auto'"),
                        ", _redis_password{}{}".format(
                            c.magenta("="),
                            c.yellow("'" + redis_password + "'"))
                        if redis_password else "")
            cli_logger.newline()
            cli_logger.print(
                cf.underlined("If connection fails, check your "
                              "firewall settings and "
                              "network configuration."))
            cli_logger.newline()
            cli_logger.print("To terminate the Ray runtime, run")
            cli_logger.print(cf.bold("  ray stop"))
    else:
        # Start Ray on a non-head node.
        if not (port is None):
            cli_logger.abort("`{}` should not be specified without `{}`.",
                             cf.bold("--port"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --port is not "
                            "allowed.")
        if redis_shard_ports is not None:
            cli_logger.abort("`{}` should not be specified without `{}`.",
                             cf.bold("--redis-shard-ports"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --redis-shard-ports "
                            "is not allowed.")
        if redis_address is None:
            cli_logger.abort("`{}` is required unless starting with `{}`.",
                             cf.bold("--address"), cf.bold("--head"))

            raise Exception("If --head is not passed in, --address must "
                            "be provided.")
        if include_dashboard:
            cli_logger.abort("`{}` should not be specified without `{}`.",
                             cf.bold("--include-dashboard"), cf.bold("--head"))

            raise ValueError(
                "If --head is not passed in, the --include-dashboard"
                "flag is not relevant.")

        # Wait for the Redis server to be started. And throw an exception if we
        # can't connect to it.
        services.wait_for_redis_to_start(
            redis_address_ip, redis_address_port, password=redis_password)

        # Create a Redis client.
        redis_client = services.create_redis_client(
            redis_address, password=redis_password)

        # Check that the version information on this node matches the version
        # information that the cluster was started with.
        if not SKIP_VERSION_CHECK:
            services.check_version_info(redis_client)

        # Get the node IP address if one is not provided.
        ray_params.update_if_absent(
            node_ip_address=services.get_node_ip_address(redis_address))

        cli_logger.labeled_value("Local node IP", ray_params.node_ip_address)

        # Check that there aren't already Redis clients with the same IP
        # address connected with this Redis instance. This raises an exception
        # if the Redis server already has clients on this node.
        check_no_existing_redis_clients(ray_params.node_ip_address,
                                        redis_client)
        ray_params.update(redis_address=redis_address)
        node = ray.node.Node(
            ray_params, head=False, shutdown_at_exit=block, spawn_reaper=block)

        cli_logger.newline()
        startup_msg = "Ray runtime started."
        cli_logger.success("-" * len(startup_msg))
        cli_logger.success(startup_msg)
        cli_logger.success("-" * len(startup_msg))
        cli_logger.newline()
        cli_logger.print("To terminate the Ray runtime, run")
        cli_logger.print(cf.bold("  ray stop"))

    if block:
        cli_logger.newline()
        with cli_logger.group(cf.bold("--block")):
            cli_logger.print(
                "This command will now block until terminated by a signal.")
            cli_logger.print(
                "Runing subprocesses are monitored and a message will be "
                "printed if any of them terminate unexpectedly.")

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
                            _tags={"exit code": str(process.returncode)})

                # shutdown_at_exit will handle cleanup.
                cli_logger.newline()
                cli_logger.error("Remaining processes will be killed.")
                sys.exit(1)
