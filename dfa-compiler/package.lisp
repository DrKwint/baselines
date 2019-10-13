;;;; package.lisp

(defpackage #:dfa-compiler
  (:use #:cl)
  (:export #:compiler-temp-hello-world))

(defpackage #:dfa-compiler.representations.dfa
  (:use #:cl)
  (:nicknames #:dfa))

(defpackage #:dfa-compiler.representations.nfa
  (:use #:cl)
  (:nicknames #:nfa))

(defpackage #:dfa-compiler.representations.regexp
  (:use #:cl)
  (:nicknames #:regexp))

(defpackage #:dfa-compiler.read-regexp
  (:use #:cl
        #:regexp))

(defpackage #:dfa-compiler.conversions.regexp-nfa
  (:use #:cl
        #:regexp
        #:nfa))

(defpackage #:dfa-compiler.conversions.nfa-dfa
  (:use #:cl
        #:nfa
        #:dfa))

(defpackage #:dfa-compiler.dfa-minimizer
  (:use #:cl
        #:dfa))

