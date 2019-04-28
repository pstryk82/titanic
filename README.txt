docker-compose up -d
docker exec -ti titanic_cli bash

python app/classify.py
