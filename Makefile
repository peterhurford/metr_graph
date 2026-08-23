.PHONY: app tests tests-serial tests-full

app:
	streamlit run visualize_projection.py

tests:
	pytest -v

# Serial — use when you need --pdb or live output from one test.
tests-serial:
	pytest -v -n0

# Production sample count (5000 trajectories) instead of the test default.
tests-full:
	_VP_SAMPLES=5000 pytest -v
