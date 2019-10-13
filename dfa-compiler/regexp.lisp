;;;; regexp.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.representations.regexp)

;;; "dfa-compiler.representations.regexp" goes here.

(defclass <regexp> () ())

(defclass <regexp-empty-string> (<regexp>) ())

(defclass <regexp-symbol> (<regexp>)
  ((symbol :initarg :symbol
           :accessor regexp-value)))

(defclass <regexp-concat> (<regexp>)
  ((sub-regexps :initarg :parts
                :type '(list <regexp>)
                :accessor regexp-value)))

(defclass <regexp-alternation> (<regexp>)
  ((sub-regexps :initarg :options
                :type '(list <regexp>)
                :accessor regexp-value)))

(defclass <regexp-maybe> (<regexp>)
  ((regexp :initarg :regexp
           :type '<regexp>
           :accessor regexp-value)))

(defclass <regexp-zero-or-more> (<regexp>)
  ((regexp :initarg :regexp
           :type '<regexp>
           :accessor regexp-value)))

;;; End dfa-compiler.representations.regexp
