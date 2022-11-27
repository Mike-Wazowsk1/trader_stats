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
CMD [ "flask", "run","--host","0.0.0.0","--port","5000"]