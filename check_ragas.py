try:
    from ragas.metrics import context_utilization
    print("context_utilization available")
except ImportError:
    print("context_utilization NOT available")

try:
    from ragas.metrics import AspectCritique
    print("AspectCritique available")
except ImportError:
    print("AspectCritique NOT available")

try:
    from ragas.metrics.critique import AspectCritique
    print("AspectCritique available in critique submodule")
except ImportError:
    print("AspectCritique NOT available in critique submodule")
