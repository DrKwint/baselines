;;;; nfa.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.representations.nfa)

;;; "dfa-compiler.representations.nfa" goes here.

(defclass <nfa-node> ()
  ((label :initarg :label)))

(defclass <nfa-edge> ()
  ((from-node :initarg :from-node
              :type <nfa-node>)
   (to-node :initarg :to-node
            :type <nfa-node>)))

(defclass <nfa-re-edge> (<nfa-edge>)
  ((regexp :initarg :regexp
           :type <regexp>)))

(defclass <nfa-symbol-edge> (<nfa-edge>)
  ((symbol :initarg :symbol)))

(defclass <nfa-symbol-epsilon> (<nfa-edge>) ())

(defclass <nfa> ()
  ((start :initform (make-instance '<nfa-node>)
          :initarg :start-node
          :type <nfa-node>)
   (accept :initform (make-instance '<nfa-node>)
           :initarg :end-node
           :type <nfa-node>)
   (nodes :initform '(list)
          :type list)
   (edges :initarg :edges
          :initform (list)
          :type list)))

(defmethod initialize-instance :after ((nfa <nfa>) &rest initargs &key &allow-other-keys)
  (declare (ignorable initargs))
  (with-slots (start accept nodes edges) nfa
    (pushnew start nodes :test #'equal)
    (pushnew accept nodes :test #'equal)
    (dolist (edge edges)
      (with-slots (from-node to-node) edge
        (pushnew from-node nodes :test #'equal)
        (pushnew to-node nodes :test #'equal)))))

(defun make-re-nfa (regexp)
  (let* ((start-node (make-instance '<nfa-node>))
         (accept-node (make-instance '<nfa-node>))
         (edge (make-instance '<nfa-re-edge>
                              :from-node start-node
                              :to-node start-node
                              :regexp regexp)))
    (make-instance '<nfa>
                   :start-node start-node
                   :end-node end-node
                   :edges (list edge))))

;;; End dfa-compiler.representations.nfa
