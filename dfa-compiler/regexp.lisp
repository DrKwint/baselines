;;;; regexp.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.representations.regexp)

;;; "dfa-compiler.representations.regexp" goes here.

(defvar *regexp-as-latex* nil)

(defclass <regexp> () ())

(defclass <regexp-empty-string> (<regexp>) ())

(defmethod print-object ((regexp <regexp-empty-string>) stream)
  (if *regexp-as-latex*
      (format stream "\\epsilon")
      (format stream "ε")))

(defclass <regexp-symbol> (<regexp>)
  ((symbol :initarg :symbol
           :accessor regexp-value)))

(defmethod print-object ((regexp <regexp-symbol>) stream)
  (format stream "~a" (regexp-value regexp)))

(defclass <regexp-complex> (<regexp>) ())

(defmethod print-object :around ((regexp <regexp-complex>) stream)
  (format stream "(")
  (call-next-method regexp stream)
  (format stream ")"))

(defclass <regexp-concat> (<regexp-complex>)
  ((sub-regexps :initarg :parts
                :type '(list <regexp>)
                :accessor regexp-value)))

(defmethod print-object ((regexp <regexp-concat>) stream)
  (mapcar #'(lambda (obj) (format stream "~a" obj))
          (regexp-value regexp)))

(defclass <regexp-alternation> (<regexp-complex>)
  ((sub-regexps :initarg :options
                :type '(list <regexp>)
                :accessor regexp-value)))

(defmethod print-object ((regexp <regexp-alternation>) stream)
  (format stream
          (if *regexp-as-latex*
              "~{~A~^ \\cup ~}"
              "~{~A~^ ∪ ~}")
          (regexp-value regexp)))

(defclass <regexp-maybe> (<regexp>)
  ((regexp :initarg :regexp
           :type '<regexp>
           :accessor regexp-value)))

(defmethod print-object ((regexp <regexp-maybe>) stream)
  (format stream "~a?" (regexp-value regexp)))

(defclass <regexp-zero-or-more> (<regexp>)
  ((regexp :initarg :regexp
           :type '<regexp>
           :accessor regexp-value)))

(defmethod print-object ((regexp <regexp-zero-or-more>) stream)
  (format stream "~a*" (regexp-value regexp)))

;;; End dfa-compiler.representations.regexp
