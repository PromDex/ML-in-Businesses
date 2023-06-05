# Базовый образ Python, на основе которого будет создан контейнер.
FROM python:3.10
# Устанавливаем переменную окружения для отключения вывода буфера при использовании Python в контейнере
ENV PYTHONUNBUFFERED 1
# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app
# Копируем файл requirements.txt в контейнер
COPY requirements.txt /app/
# Устанавливаем зависимости из файла requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Копируем все остальные файлы проекта в контейнер
COPY . /app/
# Открываем порт, который будет доступен извне контейнера
EXPOSE 8181
# Запускаем команду или приложение при старте контейнера
CMD ["python", "app/run_server.py"]




#LABEL maintainer=""
#COPY . /app
#WORKDIR /app
#RUN pip install -r requirements.txt
#EXPOSE 8181
#VOLUME /app/app/models
#COPY ./docker-entrypoint.sh /
#RUN chmod +x /docker-entrypoint.sh
#ENTRYPOINT ["sh", "docker-entrypoint.sh"]









