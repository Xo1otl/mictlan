from dataclasses import dataclass
from typing import Never

import grpc

from funsearch_worker import pb, propose

from ._handle import ObserveRequest, handle_observe
from ._llm import LLM
from ._prompt_template import Program, PromptTemplate


@dataclass
class GRPCServicer(pb.FUNSEARCHServicer):
    handle_propose: propose.HandlerFunc[Program]

    def propose(self, request: pb.ProposeRequest, context: Never) -> pb.ProposeResponse:
        try:
            req = propose.Request[Program](
                parents=[Program(skeleton=p.hypothesis, score=p.quantitative) for p in request.parents],
                specification=request.specification,
            )
            res = self.handle_propose(req)
            return pb.ProposeResponse(hypothesises=res.hypothesises)

        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return pb.ProposeResponse()

    def observe(self, request: pb.ObserveRequest, context: Never) -> pb.ObserveResponse:
        try:
            req = ObserveRequest(skeleton=request.hypothesis)
            res = handle_observe(req)
            return pb.ObserveResponse(hypothesis=res.skeleton, quantitative=res.score)
        except ValueError as e:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return pb.ObserveResponse()


def new_grpc_servicer() -> GRPCServicer:
    handle_propose = propose.new_handler(PromptTemplate(), LLM())
    return GRPCServicer(
        handle_propose=handle_propose,
    )
