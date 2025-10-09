from typing import Dict, Any
from funsearch import datadriven


class CancellableInputConverter:
    def __init__(self, client, session_data: Dict[str, Any]):
        self.client = client
        self.session_data = session_data
        self.original_converter = datadriven.InputConverter(client)

    def convert(self, formula: str, theory_explanation: str, constants_description: str, variables_description: str, insights: str):
        if self._is_cancelled():
            raise InterruptedError("Conversion cancelled")
        return self.original_converter.convert(formula, theory_explanation, constants_description, variables_description, insights)

    def _is_cancelled(self) -> bool:
        return self.session_data.get('cancelled', False)
