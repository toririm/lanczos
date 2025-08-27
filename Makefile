CC			:= icx
DEPFLAGS	:= -MMD
CFLAGS		:= -std=c11 -Wall -O3 -qopenmp
LDFLAGS 	:= -lmkl_rt
SRCS		:= $(wildcard *.c)
OBJS		:= $(SRCS:.c=.o)
DEPS		:= $(SRCS:.c=.d)

all: main.out

main.out: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c %.d
	$(CC) $(DEPFLAGS) $(CFLAGS) -c $<

$(DEPS):

include $(DEPS)

clean:
	rm -f main.out *.o *.d

.PHONY: all clean
