import logging
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection

from . import llmsr, pb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=None))
    pb.add_FUNSEARCHServicer_to_server(llmsr.new_grpc_servicer(), server)  # pyright: ignore[reportUnknownMemberType]

    service_names = (
        pb.DESCRIPTOR.services_by_name["FUNSEARCH"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("funsearch worker gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
