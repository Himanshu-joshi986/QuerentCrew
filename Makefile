.PHONY: install seed test ui api cli eval docker

install:
	pip install -r requirements.txt && pip install -e .

seed:
	python scripts/seed_database.py

test:
	pytest

ui:
	streamlit run app/streamlit_app.py

api:
	uvicorn querentcrew.api:app --reload --port 8000

cli:
	python -m querentcrew

eval:
	python evals/run_eval.py --versions v1 v2 v3

docker:
	docker compose up --build
