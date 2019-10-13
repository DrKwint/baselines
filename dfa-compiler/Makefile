LISP ?= sbcl

dfa-compiler: dfa-compiler.asd *.lisp
	$(LISP) --load dfa-compiler.asd \
	--eval '(ql:quickload :dfa-compiler)' \
	--eval '(asdf:make :dfa-compiler)' \
	--eval '(quit)'

clean:
	$(RM) dfa-compiler
