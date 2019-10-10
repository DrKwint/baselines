LISP ?= sbcl

build:
	$(LISP) --load dfa-compiler.asd \
	--eval '(ql:quickload :dfa-compiler)' \
	--eval '(asdf:make :dfa-compiler)' \
	--eval '(quit)'
