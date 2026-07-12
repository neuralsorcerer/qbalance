

def test_final_measurement_qubits_uses_final_mapping():
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(5, 2)
    qc.h(3)
    qc.cx(3, 1)
    qc.measure(3, 0)
    qc.measure(1, 1)
    assert wl._final_measurement_qubits(qc) == [3, 1]


def test_final_measurement_qubits_falls_back_to_qubit_range():
    # The lightweight stub records a measure without clbits, so no mapping can
    # be recovered and the qubit range fallback applies.
    assert wl._final_measurement_qubits(_Circ()) == [0, 1]


def test_save_refuses_to_destroy_source_dataset(tmp_path):
    out_dir = tmp_path / "balanced"
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "qbalance_dataset.json").write_text("{}", encoding="utf-8")
    (dataset_dir / "c0.qpy").write_bytes(b"x")
    record = wl.CircuitRecord(name="c0", artifact="c0.qpy", format="qpy")
    ds = wl.CircuitDataset(dataset_dir, [record])
    bw = wl.BalancedWorkload(
        dataset=ds,
        backend_spec="b",
        selections={"c0": Strategy(spec=StrategySpec(), metrics={"depth": 1})},
        baseline_metrics={"c0": {"depth": 1}},
        objective=default_objective(),
    )

    with pytest.raises(ValueError, match="source dataset"):
        bw.save(out_dir, overwrite=True)
    # The refused overwrite must leave the source dataset untouched.
    assert (dataset_dir / "qbalance_dataset.json").exists()
    assert (dataset_dir / "c0.qpy").exists()


def test_to_download_preserves_unrelated_sibling_directory(tmp_path):
    ds = _make_dataset(tmp_path, "ds_zip")
    bw = wl.BalancedWorkload(
        dataset=ds,
        backend_spec="b",
        selections={"c0": Strategy(spec=StrategySpec(), metrics={"depth": 1})},
        baseline_metrics={"c0": {"depth": 1}},
        objective=default_objective(),
    )
    sibling = tmp_path / "bundle_dir"
    sibling.mkdir()
    marker = sibling / "keep.txt"
    marker.write_text("precious", encoding="utf-8")

    zip_path = bw.to_download(tmp_path / "bundle.zip", overwrite=True)
    assert zip_path.exists()
    assert marker.read_text(encoding="utf-8") == "precious"
    # No staging directories may be left behind.
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".bundle-")]
    assert leftovers == []


def _install_seeded_fake_provider(monkeypatch):
    pkg = types.ModuleType("qiskit.providers.fake_provider")

    class GenericBackendV2:
        def __init__(self, num_qubits, seed=None):
            self.num_qubits = num_qubits
            self.seed = seed

    pkg.GenericBackendV2 = GenericBackendV2
    monkeypatch.setitem(sys.modules, "qiskit.providers.fake_provider", pkg)


def test_fake_generic_uses_deterministic_calibration_seed(monkeypatch):
    _install_seeded_fake_provider(monkeypatch)
    default = fake.resolve("fake:generic:5")
    assert default.num_qubits == 5
    assert default.seed == 0

    seeded = fake.resolve("fake:generic:5:9")
    assert seeded.seed == 9

    with pytest.raises(QBalanceError):
        fake.resolve("fake:generic:5:not-int")
    with pytest.raises(QBalanceError):
        fake.resolve("fake:generic:5:9:extra")


def test_fake_generic_falls_back_when_seed_unsupported(monkeypatch):
    pkg = types.ModuleType("qiskit.providers.fake_provider")

    class GenericBackendV2:
        def __init__(self, num_qubits):
            self.num_qubits = num_qubits

    pkg.GenericBackendV2 = GenericBackendV2
    monkeypatch.setitem(sys.modules, "qiskit.providers.fake_provider", pkg)
    assert fake.resolve("fake:generic:7").num_qubits == 7


def test_fake_ibm_resolves_via_runtime_fake_provider(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        types.ModuleType("qiskit.providers.fake_provider"),
    )
    runtime_pkg = types.ModuleType("qiskit_ibm_runtime.fake_provider")

    class FakeThingV2:
        pass

    runtime_pkg.FakeThingV2 = FakeThingV2
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.fake_provider", runtime_pkg)

    assert isinstance(fake.resolve("fake:ibm:FakeThingV2"), FakeThingV2)
    assert isinstance(fake.resolve("fake:ibm:thing"), FakeThingV2)
    with pytest.raises(QBalanceError, match="Unknown IBM fake backend"):
        fake.resolve("fake:ibm:missing")


def test_fake_ibm_requires_runtime_when_no_factory(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "qiskit.providers.fake_provider",
        types.ModuleType("qiskit.providers.fake_provider"),
    )
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime.fake_provider", None)
    with pytest.raises(OptionalDependencyError, match="qiskit-ibm-runtime"):
        fake.resolve("fake:ibm:FakeThingV2")


class _StubTarget:
    def __init__(self, instruction_errors, qubit_properties=None):
        self._instruction_errors = instruction_errors
        self.qubit_properties = qubit_properties

    def __getitem__(self, name):
        return self._instruction_errors[name]


def test_noise_helpers_read_backendv2_target():
    target = _StubTarget(
        {
            "measure": {(0,): types.SimpleNamespace(error=0.07)},
            "cx": {(1, 0): types.SimpleNamespace(error=0.02)},
        },
        qubit_properties=[types.SimpleNamespace(t1=11.0, t2=22.0)],
    )
    backend = types.SimpleNamespace(target=target)

    assert nal._safe_get_qubit_readout_error(backend, 0) == 0.07
    assert nal._safe_get_t1(backend, 0) == 11.0
    assert nal._safe_get_t2(backend, 0) == 22.0
    # Reversed direction must be found when only one direction is calibrated.
    assert nal._safe_get_2q_error(backend, "cx", 0, 1) == 0.02
    assert nal._safe_get_2q_error(backend, "cx", 1, 0) == 0.02
    assert nal._safe_get_qubit_readout_error(backend, 3) is None


def test_estimate_circuit_error_ignores_two_qubit_barriers():
    backend = types.SimpleNamespace()

    barrier_only = _Circ()
    barrier_only.data = [(_I("barrier"), [_Q(0), _Q(1)], [])]
    assert nal.estimate_circuit_error(backend, barrier_only) == 0.0

    with_gate = _Circ()
    with_gate.data = [
        (_I("barrier"), [_Q(0), _Q(1)], []),
        (_I("cx"), [_Q(0), _Q(1)], []),
    ]
    error = nal.estimate_circuit_error(backend, with_gate)
    assert math.isclose(error, 0.01)


def test_two_qubit_ops_metric_excludes_barriers():
    circ = _Circ()
    circ.data = [
        (_I("barrier"), [_Q(0), _Q(1)], []),
        (_I("cx"), [_Q(0), _Q(1)], []),
        (_I("measure"), [_Q(0)], []),
    ]
    metrics = extract_circuit_metrics(circ)
    assert metrics["two_qubit_ops"] == 1.0


def test_report_sort_value_orders_nan_last():
    assert sort_value(3) == 3.0
    assert sort_value(float("nan")) == float("inf")
    assert sort_value(None) == float("inf")
    assert sort_value("bad") == float("inf")
    values = [float("nan"), 2.0, 1.0]
    assert sorted(values, key=sort_value)[:2] == [1.0, 2.0]
