BIKESHED ?= bikeshed
BS_FILE = cpp_jit.bs
HTML_FILE = P3333R0_cpp_jit.html

.DEFAULT_GOAL := $(HTML_FILE)

$(HTML_FILE): $(BS_FILE)
	$(BIKESHED) spec $(BS_FILE) $(HTML_FILE)

clean:
	rm -f $(HTML_FILE)

.PHONY: clean
