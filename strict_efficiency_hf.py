"""Harness adapter with explicit checkpoint loading-key admission.

Same HFLM loading/inference path; only request its HF loading diagnostics and
reject incomplete loading. Does not replace the author's architecture.
"""
from unittest.mock import patch
from lm_eval.models.huggingface import HFLM
from run_efficiency_evaluation import verify_loading_info


class StrictHFLM(HFLM):
    def _create_model(self, *args, **kwargs):
        loader = self.AUTO_MODEL_CLASS.from_pretrained

        def checked(*positional, **keyword):
            keyword['output_loading_info'] = True
            model, info = loader(*positional, **keyword)
            verify_loading_info(info)
            self.efficiency_loading_info = info
            return model

        with patch.object(self.AUTO_MODEL_CLASS, 'from_pretrained', side_effect=checked):
            return super()._create_model(*args, **kwargs)
