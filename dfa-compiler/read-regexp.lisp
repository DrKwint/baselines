;;;; read-regexp.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:read-regexp)

;;; "read-regexp" goes here.

(defrule regexp
    (or maybe zero-or-more regexp-symbol)
  (:identity t))

(defrule regexp-symbol
    (+ (or #\0 #\1 #\2 #\3 #\4 #\5 #\6 #\7 #\8 #\9))
  (:text t)
  (:lambda (re-symbol)
    (make-instance '<regexp-symbol>
                   :symbol (parse-number re-symbol))))

(defrule maybe
    (and regexp #\?)
  (:destructure (re _)
                (declare (ignorable _))
                (make-instance '<regexp-maybe>
                               :regexp re)))

(defrule zero-or-more
    (and regexp #\*)
  (:destructure (re _)
                (declare (ignorable _))
                (make-instance '<regexp-zero-or-more>
                               :regexp re)))

(defrule complex
    (and #\( (or alternation) #\)))

(defrule alternation
    alternation-internal
  (:lambda (alt-list)
    (make-instance '<regexp-alternation>
                   :options alt-list)))

(defrule alternation-internal
    (or regexp
       (and regexp #\| alternation))
  (:lambda (out)
    (if (listp out)
        (cons (second out) (third out))
        (list out))))

;;; End read-regexp
