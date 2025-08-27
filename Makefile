CC			:= icx
CFLAGS		:= -std=c11 -Wall -O3 -qopenmp
LDFLAGS 	:= -lmkl_rt
SRCS		:= $(wildcard *.c)
OUTDIR		:= build
OBJS		:= $(addprefix $(OUTDIR)/, $(SRCS:.c=.o))
DEPS		:= $(addprefix $(OUTDIR)/, $(SRCS:.c=.d))
TARGET		:= main.out

DEPFLAGS	 = -MT $@ -MMD -MF $(OUTDIR)/$*.d

all: $(OUTDIR)/$(TARGET)

$(OUTDIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OUTDIR)/%.o: %.c $(OUTDIR)/%.d | $(OUTDIR)
	$(CC) $(DEPFLAGS) $(CFLAGS) -c $< -o $@

$(DEPS):

include $(DEPS)

$(OUTDIR):
	mkdir $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
