FROM ubuntu
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y texlive-latex-extra
RUN apt-get install -y python3.10 python3.10-dev
RUN apt install -y python3-pip
WORKDIR /app/
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
RUN export FLASK_APP=app.py
CMD [ "flask", "run","-h","95-163-235-201","-p","5000"]

