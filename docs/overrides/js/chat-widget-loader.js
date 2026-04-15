(function () {
  var widgetLoaded = false;

  function isWebknossosPage() {
    return window.location.pathname.startsWith('/webknossos/');
  }

  function loadWidget() {
    widgetLoaded = true;
    window.ChatWidgetConfig = {
      webhook: {
        url: 'https://docs.webknossos.org/webknossos/ask',
        route: 'general'
      },
      branding: {
        logo: 'https://static.webknossos.org/mails/email-footer-webknossos.png',
        name: 'WEBKNOSSOS',
        welcomeText: 'Hi 👋, how can we help?',
        responseTimeText: 'We typically respond right away'
      },
      style: {
        primaryColor: '#5660ff',
        secondaryColor: '#6b3fd4',
        position: 'right',
        backgroundColor: '#ffffff',
        fontColor: '#333333'
      }
    };
    var s = document.createElement('script');
    s.src = new URL('js/chat-widget.js', document.location.origin).href;
    document.head.appendChild(s);
  }

  function updateWidget() {
    var shouldShow = isWebknossosPage();
    if (shouldShow && !widgetLoaded) {
      loadWidget();
      return;
    }
    // Adjust selector to match the root element injected by chat-widget.js
    var widget = document.querySelector('.n8n-chat-widget');
    if (widget) {
      widget.style.display = shouldShow ? '' : 'none';
    }
  }

  // Intercept history.pushState and history.replaceState for SPA navigation
  function wrap(method) {
    var original = history[method];
    history[method] = function () {
      original.apply(this, arguments);
      updateWidget();
    };
  }
  wrap('pushState');
  wrap('replaceState');

  // Browser back/forward navigation
  window.addEventListener('popstate', updateWidget);

  // Initial page load
  updateWidget();
})();
