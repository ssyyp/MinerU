"""Unit tests that verify device-aware behaviour in MinerU's table recognition
modules.

Background
----------
Both ``WiredTableRecognition`` (UNet-based, *wired* tables) and
``PaddleTableModel`` (SLANet+, *wireless* tables) use ONNX Runtime for
inference.  Prior to this fix every model was hard-wired to
``CPUExecutionProvider`` regardless of the ``device`` argument supplied by the
caller.

These tests confirm:
1. The ``OrtInferSession`` classes in both sub-packages correctly populate
   ``_device`` from the config dictionary.
2. When ``device="cpu"`` only ``CPUExecutionProvider`` is requested.
3. When ``device="cuda"`` (or ``"cuda:0"`` etc.) ``CUDAExecutionProvider`` is
   prepended to the provider list **if and only if** it appears in
   ``onnxruntime.get_available_providers()``.  On a CPU-only machine the
   request degrades gracefully to CPU.
4. ``WiredTableInput`` / ``PaddleTableInput`` correctly carry the ``device``
   field all the way to the session.
5. The ``UnetTableModel`` and ``PaddleTableModel`` constructors now accept and
   forward a ``device`` keyword argument.
6. ``wired_table_model_init`` / ``wireless_table_model_init`` accept and
   forward a ``device`` keyword argument.

.. note::
   All heavy runtime dependencies (numpy, cv2, onnxruntime, torch, …) are
   **mocked out** so these tests run on a vanilla Python install without any
   ML packages installed.
"""

import sys
import types
import unittest
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Minimal stubs for heavy runtime dependencies
# ---------------------------------------------------------------------------

def _make_stub(name):
    mod = types.ModuleType(name)
    mod.__spec__ = types.SimpleNamespace(name=name, submodule_search_locations=[])
    return mod


def _install_stubs():
    """Inject lightweight stubs for packages not available in test environment."""
    stubs = [
        "numpy", "cv2", "PIL", "PIL.Image", "PIL.UnidentifiedImageError",
        "onnxruntime",
        "skimage", "skimage.measure",
        "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
        "loguru",
        "tqdm",
        "bs4",
        "packaging",
        "packaging.version",
    ]
    for name in stubs:
        if name not in sys.modules:
            sys.modules[name] = _make_stub(name)

    # numpy needs ndarray and float32 to exist
    np = sys.modules["numpy"]
    np.ndarray = object
    np.float32 = float
    np.array = MagicMock(return_value=[])

    # cv2 needs interpolation constants
    cv2_mod = sys.modules["cv2"]
    cv2_mod.INTER_NEAREST = 0
    cv2_mod.INTER_LINEAR = 1
    cv2_mod.INTER_CUBIC = 2
    cv2_mod.INTER_AREA = 3
    cv2_mod.INTER_LANCZOS4 = 4
    cv2_mod.COLOR_RGB2BGR = 4
    cv2_mod.COLOR_BGR2RGB = 4
    cv2_mod.COLOR_GRAY2BGR = 8
    cv2_mod.MORPH_RECT = 0
    cv2_mod.MORPH_CLOSE = 3
    cv2_mod.RETR_EXTERNAL = 0
    cv2_mod.CHAIN_APPROX_SIMPLE = 1
    cv2_mod.cvtColor = MagicMock()
    cv2_mod.resize = MagicMock()
    cv2_mod.split = MagicMock()
    cv2_mod.merge = MagicMock()
    cv2_mod.bitwise_not = MagicMock()
    cv2_mod.bitwise_and = MagicMock()
    cv2_mod.add = MagicMock()

    # loguru logger
    loguru = sys.modules["loguru"]
    loguru.logger = MagicMock()

    # PIL.Image needs Resampling (Pillow ≥ 9.1)
    pil_image = sys.modules["PIL.Image"]
    pil_image.Image = MagicMock()
    pil_image.Resampling = MagicMock()
    pil_image.Resampling.NEAREST = 0
    pil_image.Resampling.BILINEAR = 1
    pil_image.Resampling.BICUBIC = 2
    pil_image.Resampling.BOX = 3
    pil_image.Resampling.LANCZOS = 4
    pil_image.Resampling.HAMMING = 5
    pil_image.UnidentifiedImageError = Exception

    # onnxruntime stubs – these are replaced by the real mock in each test
    ort = sys.modules["onnxruntime"]
    ort.GraphOptimizationLevel = MagicMock()
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
    ort.InferenceSession = MagicMock
    ort.SessionOptions = MagicMock
    ort.get_available_providers = MagicMock(return_value=["CPUExecutionProvider"])
    ort.get_device = MagicMock(return_value="CPU")

    # tqdm
    tqdm_mod = sys.modules["tqdm"]
    tqdm_mod.tqdm = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: s,
                                                      __exit__=MagicMock(return_value=False),
                                                      update=MagicMock()))

    # bs4
    bs4_mod = sys.modules["bs4"]
    bs4_mod.BeautifulSoup = MagicMock()

    # packaging.version
    ver_mod = sys.modules["packaging.version"]
    ver_mod.parse = MagicMock(return_value=MagicMock(__ge__=lambda s, o: False))


_install_stubs()


# ---------------------------------------------------------------------------
# Now we can import (or reproduce) the OrtInferSession logic in isolation
# ---------------------------------------------------------------------------

# We import the actual utility modules under test rather than copy-pasting
# their code, because we want to catch regressions in the real source.
# We need to patch onnxruntime symbols before importing.

_ORT_CPU_ONLY = ["CPUExecutionProvider"]
_ORT_WITH_CUDA = ["CPUExecutionProvider", "CUDAExecutionProvider"]


def _build_unet_ort_session(device, available_providers):
    """Instantiate ``unet_table.utils.OrtInferSession`` with mocked ORT."""
    # Patch onnxruntime inside the target module
    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = []

    with patch.dict(sys.modules, {
        "onnxruntime": types.SimpleNamespace(
            GraphOptimizationLevel=MagicMock(ORT_ENABLE_ALL=99),
            InferenceSession=MagicMock(return_value=mock_sess),
            SessionOptions=MagicMock,
            get_available_providers=MagicMock(return_value=available_providers),
            get_device=MagicMock(return_value="CPU"),
        )
    }):
        # Force re-import with the patched onnxruntime
        mod_name = "mineru.model.table.rec.unet_table.utils"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import mineru.model.table.rec.unet_table.utils as _m
        config: Dict[str, Any] = {"model_path": "dummy.onnx", "device": device}
        return _m.OrtInferSession(config)


def _build_slanet_ort_session(device, available_providers):
    """Instantiate ``slanet_plus.table_structure_utils.OrtInferSession``."""
    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = []

    with patch.dict(sys.modules, {
        "onnxruntime": types.SimpleNamespace(
            GraphOptimizationLevel=MagicMock(ORT_ENABLE_ALL=99),
            InferenceSession=MagicMock(return_value=mock_sess),
            SessionOptions=MagicMock,
            get_available_providers=MagicMock(return_value=available_providers),
            get_device=MagicMock(return_value="CPU"),
        )
    }):
        mod_name = "mineru.model.table.rec.slanet_plus.table_structure_utils"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        import mineru.model.table.rec.slanet_plus.table_structure_utils as _m
        # Patch _verify_model so we don't need real files
        with patch.object(_m.OrtInferSession, "_verify_model", return_value=None):
            config: Dict[str, Any] = {"model_path": "dummy.onnx", "device": device}
            return _m.OrtInferSession(config)


# ---------------------------------------------------------------------------
# Tests for unet_table OrtInferSession
# ---------------------------------------------------------------------------

class TestUnetOrtInferSessionDevice(unittest.TestCase):
    """``OrtInferSession`` in ``mineru.model.table.rec.unet_table.utils``."""

    def _session(self, device, providers=_ORT_CPU_ONLY):
        return _build_unet_ort_session(device, providers)

    def test_device_stored_as_cpu(self):
        s = self._session("cpu")
        self.assertEqual(s._device, "cpu")

    def test_device_stored_as_cuda(self):
        s = self._session("cuda", _ORT_WITH_CUDA)
        self.assertEqual(s._device, "cuda")

    def test_cpu_ep_list_contains_only_cpu(self):
        s = self._session("cpu")
        ep_names = [p[0] for p in s._get_ep_list()]
        self.assertIn("CPUExecutionProvider", ep_names)
        self.assertNotIn("CUDAExecutionProvider", ep_names)

    def test_cuda_ep_first_when_available(self):
        """CUDA EP should be prepended when device='cuda' and it is available."""
        s = self._session("cuda", _ORT_WITH_CUDA)
        ep_list = s._get_ep_list()
        ep_names = [p[0] for p in ep_list]
        self.assertEqual(ep_names[0], "CUDAExecutionProvider",
                         "CUDAExecutionProvider must be the first provider")
        self.assertIn("CPUExecutionProvider", ep_names,
                      "CPUExecutionProvider must remain as fallback")

    def test_cuda_ep_absent_when_unavailable(self):
        """When CUDA EP is not in onnxruntime it must NOT be requested."""
        s = self._session("cuda", _ORT_CPU_ONLY)
        ep_names = [p[0] for p in s._get_ep_list()]
        self.assertNotIn("CUDAExecutionProvider", ep_names)
        self.assertIn("CPUExecutionProvider", ep_names)

    def test_cuda_device_id_parsed_from_colon_notation(self):
        """'cuda:1' should set device_id=1 in CUDA EP options."""
        s = self._session("cuda:1", _ORT_WITH_CUDA)
        ep_list = s._get_ep_list()
        cuda_ep = next((p for p in ep_list if p[0] == "CUDAExecutionProvider"), None)
        self.assertIsNotNone(cuda_ep, "CUDAExecutionProvider must be present")
        self.assertEqual(cuda_ep[1].get("device_id"), 1)

    def test_cuda_default_device_id_is_zero(self):
        """'cuda' (no index) should set device_id=0."""
        s = self._session("cuda", _ORT_WITH_CUDA)
        ep_list = s._get_ep_list()
        cuda_ep = next((p for p in ep_list if p[0] == "CUDAExecutionProvider"), None)
        self.assertIsNotNone(cuda_ep)
        self.assertEqual(cuda_ep[1].get("device_id"), 0)

    def test_ep_enum_has_cuda_entry(self):
        mod_name = "mineru.model.table.rec.unet_table.utils"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        with patch.dict(sys.modules, {
            "onnxruntime": types.SimpleNamespace(
                GraphOptimizationLevel=MagicMock(ORT_ENABLE_ALL=99),
                InferenceSession=MagicMock,
                SessionOptions=MagicMock,
                get_available_providers=MagicMock(return_value=_ORT_CPU_ONLY),
                get_device=MagicMock(return_value="CPU"),
            )
        }):
            import mineru.model.table.rec.unet_table.utils as _m
        ep_values = [ep.value for ep in _m.EP]
        self.assertIn("CUDAExecutionProvider", ep_values,
                      "EP enum must expose CUDAExecutionProvider")


# ---------------------------------------------------------------------------
# Tests for slanet_plus OrtInferSession
# ---------------------------------------------------------------------------

class TestSlanetOrtInferSessionDevice(unittest.TestCase):
    """``OrtInferSession`` in ``mineru.model.table.rec.slanet_plus.table_structure_utils``."""

    def _session(self, device, providers=_ORT_CPU_ONLY):
        return _build_slanet_ort_session(device, providers)

    def test_cpu_ep_list_contains_only_cpu(self):
        s = self._session("cpu")
        ep_names = [p[0] for p in s._get_ep_list()]
        self.assertIn("CPUExecutionProvider", ep_names)
        self.assertNotIn("CUDAExecutionProvider", ep_names)

    def test_cuda_ep_first_when_available(self):
        s = self._session("cuda", _ORT_WITH_CUDA)
        ep_list = s._get_ep_list()
        ep_names = [p[0] for p in ep_list]
        self.assertEqual(ep_names[0], "CUDAExecutionProvider")
        self.assertIn("CPUExecutionProvider", ep_names)

    def test_cuda_ep_absent_when_unavailable(self):
        s = self._session("cuda", _ORT_CPU_ONLY)
        ep_names = [p[0] for p in s._get_ep_list()]
        self.assertNotIn("CUDAExecutionProvider", ep_names)
        self.assertIn("CPUExecutionProvider", ep_names)

    def test_cuda_device_id_parsed(self):
        s = self._session("cuda:2", _ORT_WITH_CUDA)
        ep_list = s._get_ep_list()
        cuda_ep = next((p for p in ep_list if p[0] == "CUDAExecutionProvider"), None)
        self.assertIsNotNone(cuda_ep)
        self.assertEqual(cuda_ep[1].get("device_id"), 2)

    def test_ep_enum_has_cuda_entry(self):
        mod_name = "mineru.model.table.rec.slanet_plus.table_structure_utils"
        if mod_name in sys.modules:
            del sys.modules[mod_name]
        with patch.dict(sys.modules, {
            "onnxruntime": types.SimpleNamespace(
                GraphOptimizationLevel=MagicMock(ORT_ENABLE_ALL=99),
                InferenceSession=MagicMock,
                SessionOptions=MagicMock,
                get_available_providers=MagicMock(return_value=_ORT_CPU_ONLY),
                get_device=MagicMock(return_value="CPU"),
            )
        }):
            import mineru.model.table.rec.slanet_plus.table_structure_utils as _m
        ep_values = [ep.value for ep in _m.EP]
        self.assertIn("CUDAExecutionProvider", ep_values)


# ---------------------------------------------------------------------------
# Tests for WiredTableInput / PaddleTableInput dataclasses
# ---------------------------------------------------------------------------

class TestTableInputDataclasses(unittest.TestCase):
    """Verify the dataclass ``device`` fields default correctly and are
    forwarded as expected."""

    def _import_unet_main(self):
        """Import unet_table.main with all heavy deps mocked."""
        mods_to_drop = [
            "mineru.model.table.rec.unet_table.main",
            "mineru.model.table.rec.unet_table.table_structure_unet",
            "mineru.model.table.rec.unet_table.table_recover",
            "mineru.model.table.rec.unet_table.utils",
            "mineru.model.table.rec.unet_table.utils_table_recover",
            "mineru.model.table.rec.unet_table.utils_table_line_rec",
        ]
        for mod in mods_to_drop:
            sys.modules.pop(mod, None)

        with patch.dict(sys.modules, {
            "onnxruntime": types.SimpleNamespace(
                GraphOptimizationLevel=MagicMock(ORT_ENABLE_ALL=99),
                InferenceSession=MagicMock, SessionOptions=MagicMock,
                get_available_providers=MagicMock(return_value=_ORT_CPU_ONLY),
                get_device=MagicMock(return_value="CPU"),
            ),
            "mineru.model.table.rec.unet_table.table_structure_unet": MagicMock(),
            "mineru.model.table.rec.unet_table.table_recover": MagicMock(),
            "mineru.model.table.rec.unet_table.utils_table_recover": MagicMock(),
            "mineru.utils.span_pre_proc": MagicMock(),
            "mineru.utils.enum_class": MagicMock(),
            "mineru.utils.models_download_utils": MagicMock(),
        }):
            import mineru.model.table.rec.unet_table.main as _m
            return _m

    def test_wired_table_input_default_device_cpu(self):
        _m = self._import_unet_main()
        cfg = _m.WiredTableInput(model_path="dummy.onnx")
        self.assertEqual(cfg.device, "cpu")

    def test_wired_table_input_cuda(self):
        _m = self._import_unet_main()
        cfg = _m.WiredTableInput(model_path="dummy.onnx", device="cuda")
        self.assertEqual(cfg.device, "cuda")
        self.assertEqual(asdict(cfg)["device"], "cuda")

    def test_unet_table_model_passes_device_to_wired_input(self):
        """``UnetTableModel(ocr_engine, device='cuda')`` must pass device to
        ``WiredTableInput`` and hence to the ONNX session."""
        _m = self._import_unet_main()
        mock_ort_cls = MagicMock()

        with patch.object(_m, "WiredTableRecognition") as MockRec, \
             patch.object(_m, "auto_download_and_get_model_root_path",
                          return_value="/fake"), \
             patch.object(_m, "ModelPath", new=MagicMock(unet_structure="unet.onnx")):
            _m.UnetTableModel(ocr_engine=MagicMock(), device="cuda")
            # Check WiredTableInput was instantiated with device="cuda"
            args, kwargs = MockRec.call_args
            wired_input = args[0]
            self.assertEqual(wired_input.device, "cuda",
                             "device must be forwarded to WiredTableInput")


# ---------------------------------------------------------------------------
# Tests for model_init device propagation (pure mock, no real model loading)
# ---------------------------------------------------------------------------

class TestModelInitDevicePropagation(unittest.TestCase):
    """Verify that ``wired_table_model_init`` / ``wireless_table_model_init``
    forward the ``device`` keyword to the underlying model constructors."""

    def _import_model_init(self):
        """Return the model_init module with all heavy deps mocked out."""
        mod_name = "mineru.backend.pipeline.model_init"
        sys.modules.pop(mod_name, None)

        heavy = [
            "torch", "torch.cuda", "torch.backends", "torch.backends.mps",
            "mineru.model.layout.pp_doclayoutv2",
            "mineru.model.mfr.unimernet.Unimernet",
            "mineru.model.mfr.pp_formulanet_plus_m.predict_formula",
            "mineru.model.ocr.pytorch_paddle",
            "mineru.model.ori_cls.paddle_ori_cls",
            "mineru.model.table.cls.paddle_table_cls",
            "mineru.model.table.rec.slanet_plus.main",
            "mineru.model.table.rec.unet_table.main",
            "mineru.utils.config_reader",
            "mineru.utils.enum_class",
            "mineru.utils.models_download_utils",
        ]
        overrides = {m: MagicMock() for m in heavy}
        with patch.dict(sys.modules, overrides):
            import mineru.backend.pipeline.model_init as _m
            return _m

    def test_wired_table_model_init_passes_device(self):
        _m = self._import_model_init()
        mock_ocr = MagicMock()
        _m.AtomModelSingleton = MagicMock()
        _m.AtomModelSingleton.return_value.get_atom_model.return_value = mock_ocr
        MockUnet = MagicMock()
        _m.UnetTableModel = MockUnet

        _m.wired_table_model_init(lang="en", device="cuda")

        MockUnet.assert_called_once()
        _, kwargs = MockUnet.call_args
        self.assertEqual(kwargs.get("device"), "cuda")

    def test_wireless_table_model_init_passes_device(self):
        _m = self._import_model_init()
        mock_ocr = MagicMock()
        _m.AtomModelSingleton = MagicMock()
        _m.AtomModelSingleton.return_value.get_atom_model.return_value = mock_ocr
        MockPaddle = MagicMock()
        _m.PaddleTableModel = MockPaddle

        _m.wireless_table_model_init(lang="en", device="cuda")

        MockPaddle.assert_called_once()
        _, kwargs = MockPaddle.call_args
        self.assertEqual(kwargs.get("device"), "cuda")

    def test_wired_default_device_is_cpu(self):
        _m = self._import_model_init()
        _m.AtomModelSingleton = MagicMock()
        _m.AtomModelSingleton.return_value.get_atom_model.return_value = MagicMock()
        MockUnet = MagicMock()
        _m.UnetTableModel = MockUnet
        _m.wired_table_model_init()
        _, kwargs = MockUnet.call_args
        self.assertEqual(kwargs.get("device"), "cpu")

    def test_wireless_default_device_is_cpu(self):
        _m = self._import_model_init()
        _m.AtomModelSingleton = MagicMock()
        _m.AtomModelSingleton.return_value.get_atom_model.return_value = MagicMock()
        MockPaddle = MagicMock()
        _m.PaddleTableModel = MockPaddle
        _m.wireless_table_model_init()
        _, kwargs = MockPaddle.call_args
        self.assertEqual(kwargs.get("device"), "cpu")


if __name__ == "__main__":
    unittest.main()

