# Folding Collabsable Thinking function

[<Start Thinking>] 

```html/js/css
<div class="collapsible">
  <button class="collapsible-btn">Click to expand/collapse</button>
  <div class="collapsible-content">
    This is the "thinking" content. Put your reasoning or calculations here.
  </div>
</div>

<style>
.collapsible-content {
  display: none;
  padding: 10px;
  border-left: 2px solid #888;
  margin-top: 5px;
  background-color: #f9f9f9;
}
.collapsible-btn {
  cursor: pointer;
  padding: 5px 10px;
  background-color: #eee;
  border: 1px solid #ccc;
  font-weight: bold;
}
</style>

<script>
document.querySelectorAll('.collapsible-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const content = btn.nextElementSibling;
    content.style.display = content.style.display === 'block' ? 'none' : 'block';
  });
});
</script>
```

[<End Thinking>]
