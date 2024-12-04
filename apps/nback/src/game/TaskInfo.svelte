<script>
  let { n, totalQuestions } = $props();
</script>

<div class="flex justify-center gap-10 mb-5 text-lg">
  <p>n = <span class="font-bold text-blue-500">{n}</span></p>
  <p>問題数: <span class="font-bold text-blue-500">{totalQuestions}</span></p>
</div>
