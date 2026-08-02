"""Unit tests for the package's public API surface."""
import jax_es_hyperneat


def test_public_names_exported():
    from jax_es_hyperneat import JAXESHyperNEAT, TensorNEATESHyperNEATOptimized
    assert JAXESHyperNEAT is TensorNEATESHyperNEATOptimized
    assert set(jax_es_hyperneat.__all__) == {"JAXESHyperNEAT", "TensorNEATESHyperNEATOptimized"}


def test_base_class_is_the_vendored_compat_not_geenns():
    from jax_es_hyperneat import JAXESHyperNEAT
    base = JAXESHyperNEAT.__mro__[1]
    assert base.__name__ == "BaseAlgorithm"
    assert base.__module__ == "jax_es_hyperneat._compat.core.base_algorithm"
    assert "geenns" not in base.__module__


def test_impl_module_resolves_inside_package():
    import jax_es_hyperneat.eshyperneat as e
    assert "jax_es_hyperneat" in e.__file__ and "geenns.research" not in e.__file__
