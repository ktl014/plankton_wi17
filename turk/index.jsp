<%@ page contentType="text/html;charset=UTF-8" language="java" %>
<html>
  <head>
    <title>Hello World</title>
  </head>
  <body>
  <link href="https://s3.amazonaws.com/mturk-public/bs30/css/bootstrap.min.css" rel="stylesheet" />
  <section class="container" id="Other" style="margin-bottom:15px; padding: 10px 10px; font-family: Verdana, Geneva, sans-serif; color:#333333; font-size:0.9em;">
    <div class="row col-xs-12 col-md-12"><!-- Instructions -->
      <div class="panel panel-primary">
        <div class="panel-heading"><strong>Instructions</strong></div>

        <div class="panel-body">
          <p>Draw a dot at the location of head:</p>

          <ol>
            <li>Finding the Head of the copepods. Move the mouse to the location of the head.</li>
            <li>Annotate the location by clicking the left mouse. Please do NOT draw a large box. Only a small dot at the location of Head will be approved. Only one location is required, please do not draw multiple dot.</li>
            <li>A good example of annotation can be find at:<a href="https://goo.gl/3iscbB">https://goo.gl/3iscbB</a>.</li>
            <li>Head location can sometimes be hard to distinguish. To find antennas of the copepods, and then the head will be under and between the two antennas.</li>
            <li>The tail may look like antenna but tail is generally smooth while antenna has root-like structure.&nbsp;</li>
            <li>Here are some challenging images:&nbsp;<a href="https://goo.gl/H2ceCh">https://goo.gl/H2ceCh</a>. Please take a look and use as a guidance.</li>
          </ol>
        </div>
      </div>
      <!-- End Instructions --><!-- Content Body -->

      <div style="display:none;">&nbsp;</div>
    </div>
  </section>
  <link href="https://ajax.googleapis.com/ajax/libs/jqueryui/1.10.3/themes/smoothness/jquery-ui.css" rel="stylesheet" /><script src="https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script><script src="https://ajax.googleapis.com/ajax/libs/jqueryui/1.10.3/jquery-ui.js"></script>
  <div style="display:inline-block;vertical-align:top;">
    <h1>Draw a dot by clicking the mouse on the following area: ${objects_to_find}</h1>

    <p>Draw a dot using your mouse over each object that matches the search criteria &quot;${objects_to_find}&quot;. Then submit.</p>

    <%--<div id="bbox_annotator" style="display:inline-block">&nbsp;</div>--%>

    <div class="container">
      <table class="table">
        <thead>
        <tr>
          <th>#</th>
          <th>Head Position</th>
          <th>Tail Position</th>
          <th>Z Direction</th>
        </tr>
        </thead>
        <tbody>
        <tr>
            <th scope="row">ex</th>
            <td colspan="3">
                <img class="img-responsive" src="example.png">
            </td>
        </tr>
        <% for (int i = 1; i <= 8; i++) { %>
        <tr>
          <th scope="row"><%= i %></th>
          <td>
            <div class="div-images" id="div-images-head-<%= i %>">
              <img alt="Placeholder" class="img-responsive" height="200" width="200" />
            </div>
          </td>
          <td>
            <div class="div-images" id="div-images-tail-<%= i %>">
              <img alt="Placeholder" class="img-responsive" height="200" width="200" />
            </div>
          </td>
          <td>
            <form class="btn-group-vertical">
              <div class="radio"><label><input type="radio" name="specimen-radio-<%= i %>" value="Parallel">Parallel</label></div>
              <div class="radio"><label><input type="radio" name="specimen-radio-<%= i %>" value="Facing Camera">Facing Camera</label></div>
              <div class="radio"><label><input type="radio" name="specimen-radio-<%= i %>" value="Back to Camera">Back to Camera</label></div>
              <div class="radio"><label><input type="radio" name="specimen-radio-<%= i %>" value="Out of Frame">Out of Frame</label></div>
              <div class="radio"><label><input type="radio" name="specimen-radio-<%= i %>" value="Not Sure">Not Sure</label></div>
            </form>
          </td>
        </tr>
        <% } %>

        <tr>
            <td></td>
            <td></td>
            <td></td>
            <td>
                <p id="button_paragraph">
                    <input id="annotation_data" name="annotation_data" type="hidden">
                    <input id="submit_button" class="btn btn-default" type="submit">
                </p>
            </td>
        </tr>

        </tbody>
      </table>
    </div>

  <p id="result-paragraph"></p>

</div>

<script type="text/javascript">
$(document).ready(function() {
  var submission = [];
  var image_containers = $('.div-images');
  for (var i = 0; i < image_containers.length; i += 2) {
      var img = $(image_containers[i]).find('img')[0];
      submission.push({url: img.src});
  }
  for (var i = 0; i < image_containers.length; i++) {
      var container = image_containers[i];
      var img = $(container).find('img')[0];
//      img.src = urls[Math.floor(i / 2)];
      var canvas = $("<canvas/>");
      canvas.attr({
          id: 'canvas-' + i,
          class: 'img-canvas',
          width: img.width,
          height: img.height
      });
      container.prepend(canvas[0]);
  }
  var canvases = $('.img-canvas');
  var radios = $("input[type=radio]");

  canvases.on("mousedown", function(e) {
      var canvas = $(e.target);
      var n = parseInt(canvas.attr('id').slice(7)),
          idx = Math.floor(n / 2),
          label = n % 2 == 0 ? 'head' : 'tail';
      var offset = canvas.offset();
      var x = Math.round(e.pageX - offset.left),
          y = Math.round(e.pageY - offset.top);
      var ctx = canvas[0].getContext('2d');
      clear_canvas(ctx);
      draw_point(ctx, x, y, label);
      submission[idx][label] = {
          x: x,
          y: y
      };
      submission[idx].width = canvas.width();
      submission[idx].height = canvas.height();
  });

  radios.change(function(e) {
      var radio = $(e.target),
          selectedVal = radio.val();
      var idx = parseInt(radio.attr('name').slice(15)) - 1;

      submission[idx]['z-dir'] = selectedVal;
  });

  $('#submit_button').click(function(e) {
      $('#result-paragraph').text(JSON.stringify(submission));
  });
});

//    function check_submission(submission) {
//        $('#submit_button').disable();
//        if
//    }

function clear_canvas(ctx) {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
}

function draw_point(ctx, xoff, yoff, label) {
  var radius = 6;
  var pin_height = 12;
  ctx.strokeStyle = "cyan";
  ctx.lineWidth = 2;
  ctx.fillStyle = "white";
  ctx.beginPath();
  ctx.moveTo(xoff, yoff);
  ctx.lineTo(xoff, yoff - pin_height);
  ctx.stroke();
  ctx.closePath();

  ctx.beginPath();
  ctx.arc(xoff, yoff - pin_height - radius, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.closePath();

  ctx.fillStyle = "black";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "8pt sans-serif";
  ctx.fillText(label[0].toUpperCase(), xoff, yoff - pin_height - radius);
  return [xoff, yoff];
}
</script>

<style type="text/css">
  canvas {
      position: absolute;
      z-index: 1;
      cursor: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABHNCSVQICAgIfAhkiAAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAOISURBVEiJpVZPSCNnFP9NNxQKqzHQhpIIkWIy9M9CxFvGmkRTrTrgobKzQcQlpSwkp9jotlBYoQfZGlcvya2gSOLM4qmOSttgxkPOiYdtHbMHAzaHuBCzLtgubV8PG7vJMPlT%2BoOBebz3vd%2F73vve%2Bz5GPN5BKwgs7wFwF0APgFMA65IqKy0XAjC0YwTAwzDMbPXfTUSnANoieKNNgo4W8n8nEFj%2BxoiZ%2B6acPrsIBoNzFosFAGCxWBAMBuculN%2FKn7778VcCyzcNUlcpsDx7cHBw5HK5vt3f3zc6HI6rRCJRBIBEIlF0OBxXe3t7XS6Xa0lRlJzA8o6GDOLxTt1HRB9Go9E%2FrFYrJZPJcyIaISKGiBYBEBEtVuWRZDJ5brVaKRqN%2Fk5EDq0v8XinvsgCy984PDx8vLa29mYmk9mz2Wx3JFW%2BrOr%2BtZNUmQD85Pf733O5XCLHceP9%2Ff3bgod3Sqr8d8MUVSqVhVAo9MHq6uq5zWbzXztvBEmVL202253l5eVnoVDoVqVSWWhag42Njft2ux1TU1Mzkio%2Fb%2Ba8lsTv90%2F39vZCFMX7Wr2h2kQeAB2BhwEjx3EvJFX%2BUceXQkSA%2Fvn%2F2ev1Xh0dHXXF791bAXAJQJFUWWGIaP26icxmM5LJ5Nnw8PD3Wue1nVsT1GsDRflienraUiwWAQBEtCGp8l0DXrU%2FAKBUKsHn83UDeFC7WCdyD8MwdTY66AFejYpTAO5WO9DKVdKGO6j6hQHAenW2dAQCgblsNtv1rPtqURtOTVqu06XU6JhcLrcwOTmJeDz%2BCNUaAICh1tjpdH6eSqWMP0QiozqF9jAM86BBoT9Jp9NvjY2NVSRV%2FrJWUXdMZ2ZmVvL5PERR3BRYvlO7Cz0ILN%2BxtbWVODk5gSAIUa2%2BjsBkMi3F4%2FFfI5HIO4VCYasVicDynYVCQZyfn387Fos9MZlMS00JJFX%2B0%2Bv13g6Hwy85jhsXRfGpwPKjDZyPbm9vPx0YGBgPh8Mvh4aGPpNU%2BS%2BtHaN3owks%2F1E6nX4cDAbft9vtGBwcfNHX13fh8%2Fm6U6nUWTab7cpkMjfz%2BTxisdgvbrf7tqTKT%2FQC0SWokhjK5fLXm5ubkVwu17m7u4tSqQSz2YyJiQk4nc7K7OzsQ6PR%2BJ1e5C0JNGQrDMPMXctE9Eh7Whqh3StTO1WbTtlatHvpK0TUg9evCqVdgrZS9H%2FwD9NfrUmtcqn3AAAAAElFTkSuQmCC) 11 11, crosshair;
  }
</style>

</body>
</html>
