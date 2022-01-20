function objectToQuerystring (obj) {
  return Object.keys(obj).reduce(function (str, key, i) {
    var delimiter, val;
    delimiter = (i === 0) ? '?' : '&';
    key = encodeURIComponent(key);
    val = encodeURIComponent(obj[key]);
    return [str, delimiter, key, '=', val].join('');
  }, '');
}

function addressKindChange(e) {
  // 신규소스
  // const : 수정 못하는 변수로 선언 (value는 수정가능)
  const companyData = {
    game: ["엔씨소프트", "넷마블", "펄어비스"],
    realEstate: ["롯데리츠", "SK디앤디", "신한알파리츠"],
    broadCast: ["CJ ENM", "스튜디오드래곤", "SBS"],
    sea: ["HMM", "KSS해운", "와이엔텍"],
    pharmaceutical: ["셀트리온", "한미약품", "삼성바이오로직스"],
  }
  const targetType = e.value;
  const companyOptions = companyData[targetType];
  console.log('targetType', targetType)

  // 옵션 재세팅
  const targetElement = document.getElementById("company");
  targetElement.options.length = 0;

  // default options
  let defaultOption = document.createElement("option");
  defaultOption.value = "0000";
  defaultOption.innerHTML = "회사를 선택하세요";
  targetElement.appendChild(defaultOption);

  // another options
  for (option of companyOptions) {
    let element = document.createElement("option");
    element.value = option;
    element.innerHTML = option;
    targetElement.appendChild(element);
  }
}

function handleClickChoose (){
    const themeEl = document.getElementById('theme');
    const companyEl = document.getElementById('company');
    const modelEl = document.getElementById('model');
    const predictTermEl = document.getElementById('predictTerm');
    const query = {
        theme: themeEl.value,
        company: companyEl.value,
        model: modelEl.value,
        predictTerm: predictTermEl.value,
        requestStatus: 'true'
    }

    this.window.location.href =`/${objectToQuerystring(query)}`;
}

function handleClickReset(){
  const selectTagArray = document.querySelectorAll('nav#mainNav select');
  selectTagArray.forEach(item => item.value = "0000");
}
