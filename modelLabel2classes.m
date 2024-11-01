function mapind = modelLabel2classes(modelLabel,classes)
         mapind = [];
         for i = 1:numel(modelLabel)
             mapind(i) = find(classes==modelLabel(i));
         end
end