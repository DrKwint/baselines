;;;; nfa.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.representations.nfa)

;;; "dfa-compiler.representations.nfa" goes here.

(defclass <nfa-node> ()
  ((label :initarg :label)
   (edges-out :initform ())))

(defclass <nfa-edge> ()
  ((from-node :initarg :from-node
              :type <nfa-node>)
   (to-node :initarg :to-node
            :type <nfa-node>)))

(defmethod initialize-instance :after ((edge <nfa-edge>) &key &allow-other-keys)
  (with-slots (from-node) edge
    (pushnew edge (slot-value from-node 'edges-out) :test #'equal)))

(defclass <nfa-re-edge> (<nfa-edge>)
  ((regexp :initarg :regexp
           :type <regexp>)))

(defclass <nfa-symbol-edge> (<nfa-edge>)
  ((symbol :initarg :symbol)))

(defclass <nfa-epsilon-edge> (<nfa-edge>) ())

(defclass <nfa> ()
  ((start :initform (make-instance '<nfa-node>)
          :initarg :start-node
          :type <nfa-node>)
   (accept :initform (make-instance '<nfa-node>)
           :initarg :end-node
           :type <nfa-node>)
   (nodes :initform ()
          :type list)
   (edges :initarg :edges
          :initform ()
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
                              :to-node accept-node
                              :regexp regexp)))
    (make-instance '<nfa>
                   :start-node start-node
                   :end-node accept-node
                   :edges (list edge))))

(defun label-nodes (nfa)
  (let ((q (list (slot-value nfa 'start)))
        (counter 0))
    (setf (slot-value (slot-value nfa 'accept) 'label) 'accept)
    (do () ((null q))
      (let ((current (car (last q))))
        (setf q (butlast q))
        (with-slots (label edges-out) current
          (setf label (incf counter))
          (dolist (edge edges-out)
            (with-slots (to-node) edge
              (unless (slot-boundp to-node 'label)
                (pushnew to-node q :test #'equal)))))))))

(defun nfa-to-dot (filename nfa)
  (with-open-file (stream filename :direction :output
                                   :if-exists :supersede
                                   :if-does-not-exist :create)
    (format stream "digraph {~%")
    (format stream "~&~4Trankdir=LR;~%")
    (label-nodes nfa)
    (with-slots (nodes edges start accept) nfa
      (dolist (node nodes)
        (format stream "~&~4T~a;~%"
                (slot-value node 'label)))
      (dolist (edge edges)
        (with-slots ((from from-node) (to to-node)) edge
          (typecase edge
            (<nfa-symbol-edge>
             (format stream "~&~4T~a -> ~a [label = \"~a\"];~%"
                     (slot-value from 'label)
                     (slot-value to 'label)
                     (slot-value edge 'symbol)))
            (<nfa-epsilon-edge>
             (format stream "~&~4T~a -> ~a [label = \"ε\"];~%"
                     (slot-value from 'label)
                     (slot-value to 'label)))
            (<nfa-re-edge>
             (let ((*regexp-as-latex* nil))
               (format stream "~&~4T~a -> ~a [label = \"~a\"];~%"
                       (slot-value from 'label)
                       (slot-value to 'label)
                       (slot-value edge 'regexp))))))))
    (format stream "}~%")))

(defun epsilon-closure (start-node)
  (let ((process-stack (list start-node))
        (closure-set (list start-node)))
    (do ()
        ((null process-stack) closure-set)
      (let ((node (first process-stack)))
        (setf process-stack (rest process-stack))
        (dolist (next-node (mapcar #'(lambda (e) (slot-value e 'to-node))
                                   (remove-if-not #'(lambda (e) (typep e '<nfa-epsilon-edge>))
                                                  (slot-value node 'edges-out))))
          (unless (member next-node closure-set :test #'equal)
            (push next-node process-stack)
            (pushnew next-node closure-set)))))))

;;; End dfa-compiler.representations.nfa
