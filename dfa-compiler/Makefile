LISP ?= sbcl

dfa-compiler: dfa-compiler.asd *.lisp
	$(LISP) --load dfa-compiler.asd \
	--eval '(ql:quickload :dfa-compiler)' \
	--eval '(asdf:make :dfa-compiler)' \
	--eval '(quit)'

dfa-compiler.tbz2: Makefile README.md dfa-compiler.asd *.lisp
	tar cjf $@ $^

clean:
	$(RM) dfa-compiler dfa-compiler.tbz2
