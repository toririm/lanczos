CC			= icx
CFLAGS		= -std=c11 -Wall -llapack -O3 -mtune=native -march=native
SRCS		= $(wildcard *.c)
OBJS		= $(SRCS:.c=.o)

all: main.out

%.o: %.h

main.out: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -f main.out *.o

.PHONY: all clean
