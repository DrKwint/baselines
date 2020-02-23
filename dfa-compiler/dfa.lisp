;;;; dfa.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa)

;;; "dfa" goes here.

(defclass <dfa-node> ()
  ((label :initarg :label)
   (edges-out :initform ())
   (acceptp :initarg :acceptp
            :initform nil)))

(defclass <dfa-multi-node> ()
  ((involved-nfa-nodes :initarg :involved-nfa-nodes)))

(defclass <dfa-edge> ()
  ((from-node :initarg :from-node
              :type <dfa-node>)
   (to-node :initarg :to-node
            :type <dfa-node>)
   (transitionable-symbols :initarg :transitionable-symbols
                           :type list
                           :initform ())))

(defclass <dfa> ()
  ((start :initarg :start
          :type <dfa-node>)
   (nodes :initarg :nodes
          :type list
          :initform ())
   (edges :initarg :edges
          :type list
          :initform ())))


;;; End dfa
