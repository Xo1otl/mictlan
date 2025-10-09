import logging
from concurrent import futures
from typing import Never

import grpc
from grpc_reflection.v1alpha import reflection

from . import observe, pb, propose

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMSRServicer(pb.LLMSRServicer):
    def propose(self, request: pb.ProposeRequest, context: Never) -> pb.ProposeResponse:
        try:
            req = propose.Request(
                parents=[
                    propose.Program(skeleton=p.skeleton, score=p.score)
                    for p in request.parents
                ],
            )
            res = propose.handle(req)
            return pb.ProposeResponse(skeletons=res.skeletons)

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return pb.ProposeResponse()

    def observe(self, request: pb.ObserveRequest, context: Never) -> pb.ObserveResponse:
        try:
            req = observe.Request(skeleton=request.skeleton)
            res = observe.handle(req)
            return pb.ObserveResponse(skeleton=res.skeleton, score=res.score)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return pb.ObserveResponse()


def main() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=None))
    pb.add_LLMSRServicer_to_server(LLMSRServicer(), server)  # pyright: ignore[reportUnknownMemberType]

    service_names = (
        pb.DESCRIPTOR.services_by_name["LLMSR"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("llmsr worker gRPC server started on port 50051")
    server.wait_for_termination()


if __name__ == "__main__":
    main()
