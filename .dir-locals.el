;; Ask adria@far.ai for these auto-formatting functions if you want to use them
((python-mode . ((eval . (progn
                           (add-hook 'before-save-hook #'adria-python-format-buffer nil t)))))
 (js-json-mode . ((eval . (progn
                            (adria-json-on-save-mode))))))
