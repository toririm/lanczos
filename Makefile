CC			:= icx
DEPFLAGS	 = -MT $@ -MMD -MF $(OUTDIR)/$*.d
CFLAGS		:= -std=c11 -Wall -O3 -qopenmp
LDFLAGS 	:= -lmkl_rt

SRCDIR		:= src
OUTDIR		:= build

SRCS		:= $(wildcard $(SRCDIR)/*.c)
OBJS		:= $(addprefix $(OUTDIR)/, $(notdir $(SRCS:.c=.o)))
DEPS		:= $(addprefix $(OUTDIR)/, $(notdir $(SRCS:.c=.d)))
TARGET		:= main.out

all: $(OUTDIR)/$(TARGET)

$(OUTDIR)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OUTDIR)/%.o: $(SRCDIR)/%.c $(OUTDIR)/%.d | $(OUTDIR)
	$(CC) $(DEPFLAGS) $(CFLAGS) -c $< -o $@

$(DEPS):

include $(DEPS)

$(OUTDIR):
	mkdir $@

clean:
	rm -rf $(OUTDIR)

.PHONY: all clean
