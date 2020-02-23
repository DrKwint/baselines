;;;; powerset-construction.lisp
;;;;
;;;; Copyright (c) 2019 Samuel W. Flint <swflint@flintfam.org>

(in-package #:nfa-dfa)

;;; "nfa-dfa" goes here.

;; (defun subset-construction (nfa)
;;   (let* ((q0 (epsilon-closure (start-state nfa)))
;;          (q q0)
;;          (worklist (list q0)))
;;     ()))

(defun nfa-to-dfa (nfa)
  (let* ((new-start (make-instance '<dfa-multi-node> :label 'dfa-start
                                                     :involved-nfa-nodes (epsilon-closure (slot-value nfa 'start))))
         (dfa (make-instance '<dfa>
                             :start new-start)))
    ;; q_0 \gets epsilon-closure(nfa_start)
    ;; Q \gets q_0
    ;; WorkList \gets {q_0}
    ;; while (Worklist \neq NULL):
    ;;    remove q from WorkList
    ;;    For each char c in Sigma:
    ;;    t \gets epsilon-closure(Delta(Q, c))
    ;;    T[q, c] \gets t
    ;;    if t \neq Q, add t to Q and WorkList

    ))

;;; End nfa-dfa
