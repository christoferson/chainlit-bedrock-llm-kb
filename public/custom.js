window.addEventListener("chainlit-call-fn", (e) => {
    const { name, args, callback } = e.detail;
    if (name === "fetch_headers") {
        var req = new XMLHttpRequest();
        req.open('HEAD', document.location, true);
        req.send(null);
        req.onload = function() {
          var headers_str = req.getAllResponseHeaders().toLowerCase();
          //console.log(headers_str);
          const headers = {};
          headers_str
            .trim()
            .split(/[\r\n]+/)
            .map(value => value.split(/: /))
            .forEach(keyValue => {
                headers[keyValue[0].trim()] = keyValue[1].trim();
            });
          callback(headers);
        };        
    }
  });