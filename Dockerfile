FROM gcc:13
COPY task/harness.c /opt/harness.c
RUN gcc -O2 -c /opt/harness.c -o /opt/harness.o
WORKDIR /work
