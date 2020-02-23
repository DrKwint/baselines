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

(defmethod initialize-instance :after ((dfa <dfa>) &rest initargs &key &allow-other-keys)
  (declare (ignorable initargs))
  (with-slots (start nodes edges) dfa
    (pushnew start nodes :test #'equal)
    (dolist (edge edges)
      (with-slots (from-node to-node) edge
        (pushnew from-node nodes :test #'equal)
        (pushnew to-node nodes :test #'equal)))))

(defun label-nodes (dfa)
  (let ((q (list (slot-value dfa 'start)))
        (counter 0))
    (do () ((null q))
      (let ((current (car (last q))))
        (setf q (butlast q))
        (unless (slot-boundp current 'label)
          (with-slots (label edges-out acceptp) current
            (if acceptp
                (setf label (symbolicate 'accept_
                                         (format nil "~a" (incf counter))))
                (setf label (incf counter)))))
        (dolist (edge (slot-value current 'edges-out))
          (with-slots (to-node) edge
            (unless (slot-boundp to-node 'label)
              (pushnew to-node q :test #'equal))))))))

;;; End dfa
