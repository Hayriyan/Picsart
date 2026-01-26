.PHONY: help setup up down logs test

help:
	@echo "Lab Platform - Available Commands:"
	@echo "  make setup           - Install dependencies"
	@echo "  make up              - Start all services"
	@echo "  make down            - Stop all services"
	@echo "  make logs            - View logs"
	@echo "  make test            - Run tests"

setup:
	cp .env.example .env

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

test:
	docker-compose exec backend python -m pytest -v
