#lang racket

; test framework (ignore this part)

(define-syntax test-case
  (syntax-rules ()
    ((_ test-name actual expected)
     (let ((result
            (with-handlers ([exn:fail? (lambda (e) "failed")])
              (if (equal? actual expected) "OK" "failed"))))
       (printf "~a: ~a~n" test-name result)))))

; Problem 1.1: (a (b c) d ((e f) (g)))

(define p11list
  (cons 'a (cons '(b c) (cons 'd (cons (cons '(e f) '((g))) '())))))

(test-case "p1.1" p11list '(a (b c) d ((e f) (g))))

; Problem 1.2: '(* a 4) such that ((car '(* a 4)) 5 3) equals 15

(define p12list
  (cons * '(5 3)))

(test-case "p1.2" ((car p12list) 5 3) 15)

; Problem 2.1

(define flatten
  (lambda (l)
    (cond ((null? l) '())
          ((pair? l) (append (flatten (car l)) (flatten (cdr l))))
          (else (list l))
    )
  )
)
      

(test-case "p2.1"
           (flatten '(a ((b) (c d) (((e))))))
           '(a b c d e))

; Problem 2.2

(define rev
  (lambda (l)
    (cond ((null? l) l)
          ((pair? (car l))(append (rev (cdr l)) (list (rev (car l)))))
          (else (append (rev (cdr l)) (list (car l))))
    )
  )
)



(test-case "p2.2"
           (rev '(a ((b) (c d) (((e))))))
           '(((((e))) (d c) (b)) a))


; Problem 2.3

(define double
  (lambda (l)
    (cond ((null? l) l)
          ((pair? (car l)) (append
                             (list (double (car l)))
                             (double (cdr l))
                           )
          )
          (else (append (list (car l)) (append (list (car l)) (double (cdr l)))))
    )
  )
)


(test-case "p2.3"
           (double '(a ((b) (c d) (((e))))))
           '(a a ((b b) (c c d d) (((e e))))))

; Problem 2.4

(define delete
  (lambda (x l)
    (cond ((null? l) '())
          ((pair? (car l)) (append
                             (list (delete x (car l)))
                             (delete x (cdr l))))
          ((equal? x (car l)) (delete x (cdr l)))
          (else (append
                  (list (car l))
                  (delete x (cdr l))))
    )
  )
)

(delete 'c '(a ((b) (c d) (((e))))))

(test-case "p2.4 (1)"
           (delete 'c '(a ((b) (c d) (((e))))))
           '(a ((b) (d) (((e))))))

(test-case "p2.4 (2)"
           (delete 'f '(a ((b) (c d) (((e))))))
           '(a ((b) (c d) (((e))))))

; Problem 3

(define NewTable
  (lambda ()
    'YOUR-CODE-HERE))

(define InsertIntoTable
  (lambda (entry table)
    'YOUR-CODE-HERE))

(define LookupTable
  (lambda (variable table)
    'YOUR-CODE-HERE))

(define table
  (InsertIntoTable '(b (2 4 5)) (InsertIntoTable '(a 7) (NewTable))))

(test-case "p3 (1)" (LookupTable 'a table) 7)
(test-case "p3 (2)" (LookupTable 'b table) '(2 4 5))
(test-case "p3 (3)" (LookupTable 'c table) '())

; Problem 4

; map is already defined in Racket
;
;(define map
;  (lambda (f l)
;    (if (null? l)
;        '()
;        (cons (f (car l)) (map f (cdr l))))))

(define reduce
  (lambda (op l id)
    (if (null? l)
        id
        (op (car l) (reduce op (cdr l) id)))))

(define minSquareVal
  (lambda (l)
    'YOUR-CODE-HERE))

(define maxSquareVal
  (lambda (l)
    'YOUR-CODE-HERE))

(test-case "p4 (1)" (minSquareVal '(-5 3 -7 10 -11 8 7)) 9)

(test-case "p4 (2)" (maxSquareVal '(-5 3 -7 10 -11 8 7)) 121)
