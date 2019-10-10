;;;; dfa-compiler.lisp

(in-package #:dfa-compiler)

(defun compiler-temp-hello-world (&rest args)
  (format t "Hello World!~%")
  (uiop:quit 0 t))
