;;;; utils.lisp
;;;;
;;;; Copyright (c) 2020 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.utils
            )

;;; "dfa-compiler.utils" goes here.

(defun ensure-list (object-or-list)
  (if (listp object-or-list)
      object-or-lisp
      (list object-or-list)))

;;; End dfa-compiler.utils
            
