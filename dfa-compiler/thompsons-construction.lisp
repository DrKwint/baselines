;;;; thompsons-construction.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:dfa-compiler.conversions.regexp-nfa)

;;; "dfa-compiler.conversions.regexp-nfa" goes here.

(defclass <gnfa> (<nfa>)
  ((re-edge-count :initform 0)))

(defmethod update-instance-for-different-class :after ((old <nfa>) (new <gnfa>) &key &allow-other-keys)
  (with-slots (re-edge-count edges) new
    (dolist (edge edges)
      (when (typep edge '<nfa-re-edge>)
        (incf re-edge-count)))))

(defgeneric regexp-make-edges (regexp start-node end-node))

(defmethod regexp-make-edges ((regexp <regexp-symbol>) start-node end-node)
  (list (make-instance '<nfa-symbol-edge>
                       :from-node start-node
                       :to-node end-node
                       :symbol (regexp-value regexp))))

(defmethod regexp-make-edges ((regexp <regexp-concat>) start-node end-node)
  (let ((new-edges (list))
        (new-nodes (list))
        (sub-regexps (regexp-value regexp))
        (last-node start-node)
        (last-edge))
    (dolist (regex sub-regexps)
      (unless (or (null last-node) (equal last-node start-node))
        (pushnew last-node new-nodes :test #'equal))
      (unless (null last-edge)
        (pushnew last-edge new-edges :test #'equal))
      (let ((new-node (make-instance '<nfa-node>)))
        (setf last-edge (make-instance '<nfa-re-edge>
                                       :from-node last-node
                                       :to-node new-node
                                       :regexp regex)
              last-node new-node)))
    (pushnew last-edge new-edges :test #'equal)
    (setf (slot-value last-edge 'to-node) end-node)
    (values new-edges new-nodes)))

(defmethod regexp-make-edges ((regexp <regexp-alternation>) start-node end-node)
  (let ((new-edges (list)))
    (dolist (regexp-new (regexp-value regexp) new-edges)
      (pushnew (make-instance '<nfa-re-edge>
                              :from-node start-node
                              :to-node end-node
                              :regexp regexp-new)
               new-edges :test #'equal))))

(defmethod regexp-make-edges ((regexp <regexp-maybe>) start-node end-node)
  (list
   (make-instance '<nfa-epsilon-edge>
                  :from-node start-node
                  :to-node end-node)
   (make-instance '<nfa-re-edge>
                  :from-node start-node
                  :to-node end-node
                  :regexp (regexp-value regexp))))

(defmethod regexp-make-edges ((regexp <regexp-zero-or-more>) start-node end-node)
  (let ((new-edges (list))
        (new-nodes (list))
        (inner-re (regexp-value regexp))
        (i-begin-node (make-instance '<nfa-node>))
        (i-end-node (make-instance '<nfa-node>)))
    (pushnew i-begin-node new-nodes :test #'equal)
    (pushnew i-end-node new-nodes :test #'equal)
    (pushnew (make-instance '<nfa-epsilon-edge> :from-node start-node :to-node end-node)
             new-edges :test #'equal)
    (pushnew (make-instance '<nfa-epsilon-edge>
                            :from-node i-end-node
                            :to-node i-begin-node)
             new-edges :test #'equal)
    (pushnew (make-instance '<nfa-epsilon-edge>
                            :from-node start-node
                            :to-node i-begin-node)
             new-edges :test #'equal)
    (pushnew (make-instance '<nfa-epsilon-edge>
                            :from-node i-end-node
                            :to-node end-node)
             new-edges :test #'equal)
    (pushnew (make-instance '<nfa-re-edge>
                            :from-node i-begin-node
                            :to-node i-end-node
                            :regexp inner-re)
             new-edges :test #'equal)
    (values new-edges new-nodes)))

(defun nfa-conversion (regexp)
  (let ((regexp-nfa (change-class (make-re-nfa regexp) '<gnfa>)))
    (with-slots (re-edge-count edges nodes) regexp-nfa
      (do ()
          ((= re-edge-count 0) (change-class regexp-nfa '<nfa>))
        (let ((edge-process (find-if #'(lambda (e) (typep e '<nfa-re-edge>)) edges)))
          (with-slots (regexp from-node to-node) edge-process
            (setf (slot-value from-node 'edges-out) (remove edge-process (slot-value from-node 'edges-out)))
            (multiple-value-bind (new-edges new-nodes) (regexp-make-edges regexp from-node to-node)
              (decf re-edge-count)
              (setf edges (remove edge-process edges :test #'equal))
              (dolist (new-edge new-edges)
                (pushnew new-edge edges :test #'equal)
                (when (typep new-edge '<nfa-re-edge>)
                  (incf re-edge-count)))
              (dolist (new-node new-nodes)
                (pushnew new-node nodes :test #'equal)))))))))

;;; End dfa-compiler.conversions.regexp-nfa
