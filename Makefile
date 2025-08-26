CC			= icx
CFLAGS		= -std=c11 -Wall -O3 -qopenmp
LDFLAGS 	= -lmkl_rt
SRCS		= $(wildcard *.c)
OBJS		= $(SRCS:.c=.o)

all: main.out

%.o: %.h

main.out: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f main.out *.o

.PHONY: all clean
