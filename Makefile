.PHONY: app tests

app:
	streamlit run visualize_projection.py

tests:
	pytest -v
